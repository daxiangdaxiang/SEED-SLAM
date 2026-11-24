"""Persistent GTSAM local reference; not the distributed DRANPPR algorithm."""

from dataclasses import dataclass
import math
import time

import gtsam
import numpy as np


def landmark_ordering_constraints(mode, existing_landmarks, new_landmarks):
    """Return causal landmark ordering constraints for one incremental update."""
    if mode not in {'natural', 'new_last', 'all_last'}:
        raise ValueError('landmark ordering must be natural, new_last, or all_last')
    if new_landmarks is None:
        new_landmarks = set()
    else:
        try:
            new_landmarks = set(new_landmarks)
        except TypeError:
            new_landmarks = {new_landmarks}
    if mode == 'natural':
        return None
    if mode == 'new_last':
        return None if not new_landmarks else {key: 1 for key in new_landmarks}
    landmarks = set(existing_landmarks)
    landmarks.update(new_landmarks)
    return {key: 1 for key in landmarks}


@dataclass(frozen=True)
class UpdateStatistics:
    elapsed_ms: float
    relinearized: int
    reeliminated: int
    inserted: int
    removed: int


class IncrementalFactorGraph:
    def __init__(self, relinearize_threshold=1e-4, wildfire_threshold=1e-3):
        if not math.isfinite(relinearize_threshold) or relinearize_threshold <= 0:
            raise ValueError('relinearization threshold must be positive')
        if not math.isfinite(wildfire_threshold) or wildfire_threshold < 0:
            raise ValueError('wildfire threshold must be finite and nonnegative')
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.relinearizeSkip = 1
        params.findUnusedFactorSlots = True
        params.setFactorization('QR')
        optimizer = gtsam.ISAM2GaussNewtonParams()
        optimizer.setWildfireThreshold(wildfire_threshold)
        params.setOptimizationParams(optimizer)
        self.solver = gtsam.ISAM2(params)
        self.factors = {}
        self.slots = {}
        self.keys = set()
        self.failed = False

    @property
    def factor_count(self):
        return len(self.factors)

    def update(self, factors, initial=None, remove=(), constrained_keys=None):
        if self.failed:
            raise RuntimeError('previous GTSAM update failed; rebuild before reusing this graph')
        initial = gtsam.Values() if initial is None else initial
        new_keys = set(initial.keys())
        remove = tuple(remove)
        if len(set(remove)) != len(remove) or any(key not in self.slots for key in remove):
            raise ValueError('removal requires distinct existing factor IDs')
        if set(remove) & factors.keys() or self.keys & new_keys:
            raise ValueError('conflicting factor edit or duplicate variable initialization')
        for factor in factors.values():
            if factor is None or any(k not in self.keys and k not in new_keys for k in factor.keys()):
                raise ValueError('factor references a missing variable')
        constrained_keys = {} if constrained_keys is None else dict(constrained_keys)
        if any(key not in self.keys and key not in new_keys
               for key in constrained_keys):
            raise ValueError('ordering constraint references a missing variable')
        replaced = [key for key in factors if key in self.slots]
        deleted = [*remove, *replaced]
        new_graph = gtsam.NonlinearFactorGraph()
        for factor in factors.values():
            new_graph.push_back(factor)
        start = time.perf_counter()
        try:
            removed_slots = [self.slots[key] for key in deleted]
            if constrained_keys:
                groups = gtsam.KeyGroupMap()
                for key, group in constrained_keys.items():
                    groups.insert2(key, group)
                result = self.solver.update(
                    new_graph, initial, removed_slots, groups)
            else:
                result = self.solver.update(new_graph, initial, removed_slots)
        except Exception:
            # ISAM2 does not promise rollback after a failed numerical update.
            self.failed = True
            raise
        elapsed = 1000. * (time.perf_counter() - start)
        for key in deleted:
            del self.slots[key]
            del self.factors[key]
        self.slots.update(zip(factors, result.getNewFactorsIndices()))
        self.factors.update(factors)
        self.keys.update(new_keys)
        return UpdateStatistics(elapsed, result.getVariablesRelinearized(),
                                result.getVariablesReeliminated(), len(factors), len(deleted))

    def estimate(self):
        if self.failed:
            raise RuntimeError('cannot use a failed graph')
        return self.solver.calculateEstimate()

    def factor_graph(self, exclude=()):
        """Materialize only for audits/batch controls, never during each update."""
        graph = gtsam.NonlinearFactorGraph()
        for key, factor in self.factors.items():
            if key not in exclude:
                graph.push_back(factor)
        return graph


class IncrementalPoint2Robot:
    """One independent local Pose2/Point2 graph in its own fixed gauge.

    Raw polar observations are not converted to isotropic Cartesian factors.
    Odometry sigmas are per sqrt(second), a declared diffusion model rather
    than a sensor calibration. No inter-robot equality is added by this class.
    """

    def __init__(self, robot, range_sigma=.1, bearing_sigma=.05,
                 odometry_sigma=(.1, .1, .05), relinearize_threshold=1e-4,
                 wildfire_threshold=1e-3, landmark_ordering='natural'):
        if not isinstance(robot, int) or isinstance(robot, bool) or robot < 0:
            raise ValueError('robot ID must be a nonnegative integer')
        sigma = np.asarray([range_sigma, bearing_sigma, *odometry_sigma], dtype=float)
        if sigma.shape != (5,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('measurement standard deviations must be finite and positive')
        self.robot = robot
        self.graph = IncrementalFactorGraph(relinearize_threshold, wildfire_threshold)
        self.observation_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([bearing_sigma, range_sigma]))
        self.observation_variance_model = None
        self.odometry_sigma = np.asarray(odometry_sigma)
        self.pose_index = -1
        self.pose_time = None
        self.last_time = -float('inf')
        self.landmark_keys = set()
        landmark_ordering_constraints(landmark_ordering, set(), None)
        self.landmark_ordering = landmark_ordering

    def push(self, packet):
        return self.push_batch([packet])

    def push_batch(self, packets):
        """Insert one robot's timestamp-atomic sensor frame in one transaction."""
        packets = tuple(packets)
        if not packets:
            raise ValueError('sensor frame must contain at least one packet')
        t = packets[0]['timestamp_s']
        for packet in packets:
            times = [packet['timestamp_s'], packet['available_time_s'],
                     packet['support_max_time_s']]
            if (packet['robot'] != self.robot or not np.isfinite(times).all() or
                    packet['timestamp_s'] != t or t < self.last_time or
                    packet['support_max_time_s'] > packet['available_time_s'] or
                    packet['available_time_s'] != t):
                raise ValueError('expected one causal robot/timestamp sensor frame')
        pose_packets = [packet for packet in packets if packet['kind'] == 'pose']
        if len(pose_packets) > 1 or (pose_packets and packets[0] is not pose_packets[0]):
            raise ValueError('sensor frame permits at most one leading pose packet')

        initial = gtsam.Values()
        factors = {}
        new_landmarks = set()
        if pose_packets:
            packet = pose_packets[0]
            pose_index = packet['pose_index']
            if (not isinstance(pose_index, int) or isinstance(pose_index, bool) or
                    pose_index < 0):
                raise ValueError('invalid pose index')
            pose_key = gtsam.symbol('x', pose_index)
            if pose_index != self.pose_index + 1:
                raise ValueError('poses must be appended consecutively')
            if self.pose_index < 0:
                if packet['from_pose_index'] is not None or packet['delta'] is not None:
                    raise ValueError('initial pose has no parent')
                initial.insert(pose_key, gtsam.Pose2())
                factors['gauge'] = gtsam.PriorFactorPose2(
                    pose_key, gtsam.Pose2(), gtsam.noiseModel.Constrained.All(3))
                pose_estimate = gtsam.Pose2()
            else:
                dt = packet['duration_s']
                delta = np.asarray(packet['delta'], dtype=float)
                if (packet['from_pose_index'] != self.pose_index or not np.isfinite(dt) or dt <= 0 or
                        abs(t - self.pose_time - dt) > 1e-6 or
                        delta.shape != (3,) or not np.isfinite(delta).all()):
                    raise ValueError('invalid odometry increment')
                parent_key = gtsam.symbol('x', self.pose_index)
                relative = gtsam.Pose2(*delta)
                pose_estimate = self.graph.solver.calculateEstimatePose2(
                    parent_key).compose(relative)
                initial.insert(pose_key, pose_estimate)
                noise = gtsam.noiseModel.Diagonal.Sigmas(self.odometry_sigma * np.sqrt(dt))
                factors[('odometry', pose_index)] = gtsam.BetweenFactorPose2(
                    parent_key, pose_key, relative, noise)
        else:
            pose_index = self.pose_index
            if pose_index < 0 or t != self.pose_time:
                raise ValueError('observation-only frame requires the current pose')
            pose_key = gtsam.symbol('x', pose_index)
            pose_estimate = self.graph.solver.calculateEstimatePose2(pose_key)

        for packet in packets[len(pose_packets):]:
            if packet['kind'] != 'observation':
                raise ValueError('unsupported packet kind')
            if pose_index != self.pose_index or t != self.pose_time:
                if not pose_packets or packet['pose_index'] != pose_index:
                    raise ValueError('observation must refer to the sensor-frame pose')
            elif packet['pose_index'] != pose_index:
                raise ValueError('observation must refer to the sensor-frame pose')
            row, subject = packet['source_row'], packet['landmark_subject']
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in (row, subject)):
                raise ValueError('invalid source or landmark ID')
            distance, bearing = packet['range_m'], packet['bearing_rad']
            if not np.isfinite([distance, bearing]).all() or distance <= 0:
                raise ValueError('invalid polar observation')
            factor_id = ('observation', row)
            if factor_id in self.graph.factors or factor_id in factors:
                raise ValueError('observation source row was already consumed')
            landmark_key = gtsam.symbol('l', subject)
            noise = self.observation_noise
            if (self.observation_variance_model is not None and
                    landmark_key in self.graph.keys):
                point = self.graph.solver.calculateEstimatePoint2(landmark_key)
                predicted_range = float(pose_estimate.range(point))
                variance = np.asarray(
                    self.observation_variance_model(predicted_range), dtype=float)
                if (variance.shape != (2,) or not np.isfinite(variance).all() or
                        np.any(variance <= 0.)):
                    raise ValueError('conditional observation variance must be positive')
                noise = gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(variance))
            if landmark_key not in self.graph.keys and landmark_key not in new_landmarks:
                new_landmarks.add(landmark_key)
                initial.insert(landmark_key, pose_estimate.transformFrom(np.array(
                    [distance * np.cos(bearing), distance * np.sin(bearing)])))
            factors[factor_id] = gtsam.BearingRangeFactor2D(
                pose_key, landmark_key, gtsam.Rot2(bearing), distance,
                noise)
        result = self.graph.update(
            factors, initial,
            constrained_keys=landmark_ordering_constraints(
                self.landmark_ordering, self.landmark_keys, new_landmarks))
        if pose_packets:
            self.pose_index, self.pose_time = pose_index, t
        self.landmark_keys.update(new_landmarks)
        self.last_time = t
        return result

    def physical_graph(self):
        return self.graph.factor_graph(exclude=('gauge',))

    def physical_factor_items(self):
        return [(key, factor) for key, factor in self.graph.factors.items()
                if key != 'gauge']
