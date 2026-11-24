import json
from pathlib import Path
import sys

import gtsam
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
from dranppr_incremental import IncrementalPoint2Robot
from audit_online_noise import predicted_observations, replay, calibrate_robot


def initial_frame():
    return [dict(robot=1, kind='pose', timestamp_s=0., available_time_s=0.,
                 support_max_time_s=0., pose_index=0, from_pose_index=None,
                 delta=None, duration_s=0.),
            dict(robot=1, kind='observation', timestamp_s=0., available_time_s=0.,
                 support_max_time_s=0., pose_index=0, source_row=0,
                 landmark_subject=6, range_m=2., bearing_rad=.2)]


def next_frame(index, distance=2.):
    return [dict(robot=1, kind='pose', timestamp_s=float(index),
                 available_time_s=float(index), support_max_time_s=float(index),
                 pose_index=index, from_pose_index=index-1,
                 delta=[.1, 0., .02], duration_s=1.),
            dict(robot=1, kind='observation', timestamp_s=float(index),
                 available_time_s=float(index), support_max_time_s=float(index),
                 pose_index=index, source_row=index, landmark_subject=6,
                 range_m=distance, bearing_rad=.2)]


@pytest.mark.parametrize('parent_index', [0, 3])
def test_prediction_uses_full_pose_landmark_covariance_and_no_test_measurement(parent_index):
    robot = IncrementalPoint2Robot(1)
    robot.push_batch(initial_frame())
    for index in range(1, parent_index + 1):
        robot.push_batch(next_frame(index, 2. - .08 * index))
    frame = next_frame(parent_index + 1)
    output = predicted_observations(robot, frame)
    changed = predicted_observations(robot, next_frame(parent_index + 1, 50.))
    assert len(output) == 1
    np.testing.assert_allclose(output[0]['state_covariance'], changed[0]['state_covariance'])
    assert output[0]['predicted_range'] == changed[0]['predicted_range']
    assert output[0]['nll'] != changed[0]['nll']
    assert robot.pose_index == parent_index
    # Compare propagation against explicitly extending the graph by odometry only.
    graph = robot.graph.factor_graph()
    values = robot.graph.estimate()
    parent = gtsam.symbol('x', parent_index)
    new = gtsam.symbol('x', parent_index + 1)
    delta = gtsam.Pose2(*frame[0]['delta'])
    values.insert(new, values.atPose2(parent).compose(delta))
    graph.push_back(gtsam.BetweenFactorPose2(parent, new, delta,
        gtsam.noiseModel.Diagonal.Sigmas(robot.odometry_sigma)))
    marginal = gtsam.Marginals(graph, values)
    keys = [new, gtsam.symbol('l', 6)]
    joint = marginal.jointMarginalCovariance(keys)
    covariance = np.block([[joint.at(a, b) for b in keys] for a in keys])
    factor = gtsam.BearingRangeFactor2D(*keys, gtsam.Rot2(.2), 2.,
                                       gtsam.noiseModel.Unit.Create(2))
    ordering = gtsam.Ordering()
    for key in keys:
        ordering.push_back(key)
    linear = gtsam.GaussianFactorGraph()
    linear.push_back(factor.linearize(values))
    jacobian, _ = linear.jacobian(ordering)
    np.testing.assert_allclose(output[0]['state_covariance'],
                               jacobian @ covariance @ jacobian.T,
                               rtol=1e-9, atol=1e-10)


def test_chronological_replay_prefix_is_independent_of_future_measurements():
    packets = initial_frame()
    for index in range(1, 6):
        packets.extend(next_frame(index, distance=2. - .08 * index))
    changed = [dict(packet) for packet in packets]
    for packet in changed:
        if packet['timestamp_s'] > 3. and packet['kind'] == 'observation':
            packet['range_m'] += .1
    arms = ('fixed', 'profiled_diagonal')
    a = replay(packets, warmup_s=3., duration_s=5., methods=arms)
    b = replay(changed, warmup_s=3., duration_s=5., methods=arms)
    assert a['calibration'] == b['calibration']
    assert all(row['elapsed_s'] > 3. for row in a['predictions'])
    assert all(row['training_max_elapsed_s'] <= 3. for row in a['calibration'])
    assert len(a['online_poses']) == len(b['online_poses'])
    assert a['packet_count'] == 12


def test_custom_calibration_changes_only_future_odometry_and_keeps_global_clock():
    calls = []
    def calibrator(robot, method):
        calls.append((robot.robot, robot.last_time, method))
        return dict(method=method, robot=robot.robot, success=True,
                    at_numerical_boundary=False, bearing_variance=.05**2,
                    range_intercept_variance=.1**2, range_distance_squared_coefficient=0.,
                    odometry_variance_scales=[4., 9., 16.])
    packets = [dict(p, robot=2) for p in initial_frame()]
    selected = initial_frame()
    for index in range(1, 5):
        selected.extend(next_frame(index, 2.-.08*index))
    for packet in selected:
        for key in ('timestamp_s', 'available_time_s', 'support_max_time_s'):
            packet[key] += .5
    packets += selected
    result = replay(packets, 3., 5., methods=('fixed', 'probe'),
                    calibrator=calibrator, robot_ids={1})
    assert calls == [(1, 2.5, 'probe')]
    assert result['start_time_s'] == 0.
    assert {row['robot'] for row in result['online_poses']} == {1}
    fixed = [row for row in result['predictions'] if row['method'] == 'fixed'][0]
    probe = [row for row in result['predictions'] if row['method'] == 'probe'][0]
    assert probe['state_bearing_variance'] > fixed['state_bearing_variance']


@pytest.mark.parametrize('bearing_variance', [.05**2, 2.8e-10])
@pytest.mark.parametrize('sparse_jacobian', [False, True])
def test_sparse_prediction_matches_dense_at_nominal_and_collapsed_noise(bearing_variance, sparse_jacobian):
    from dranppr_sparse_prediction import sparse_selected_covariance
    from dranppr_sparse_linearization import sparse_linearize
    robot = IncrementalPoint2Robot(1)
    robot.push_batch(initial_frame())
    robot.observation_variance_model = lambda d: np.array([bearing_variance, .01])
    for index in range(1, 5):
        robot.push_batch(next_frame(index, 2.-.08*index))
    dense = predicted_observations(robot, next_frame(5))[0]
    kwargs = dict(covariance_solver=sparse_selected_covariance)
    if sparse_jacobian:
        kwargs['linearizer'] = sparse_linearize
    sparse = predicted_observations(robot, next_frame(5), **kwargs)[0]
    np.testing.assert_allclose(sparse['state_covariance'], dense['state_covariance'],
                               rtol=1e-7, atol=1e-11)
    assert sparse['nll'] == pytest.approx(dense['nll'], rel=1e-7, abs=1e-8)
