import sys
from pathlib import Path
import csv
import hashlib
import json

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_incremental import (
    IncrementalFactorGraph,
    IncrementalPoint2Robot,
    landmark_ordering_constraints,
)
from dranppr_linear_evidence import FactorDeletedBlockEvidence
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from audit_residual_evidence import (
    deleted_group_calibration_rows,
    deleted_observation_model_rows,
    linearize,
    native_audit,
)


def prior(x, sigma=1.):
    return gtsam.PriorFactorPoint2(1, np.array([x, 0.]), gtsam.noiseModel.Isotropic.Sigma(2, sigma))


def initialized():
    graph = IncrementalFactorGraph(relinearize_threshold=1e-10,
                                   wildfire_threshold=0.)
    values = gtsam.Values()
    values.insert(1, np.array([0., 0.]))
    graph.update({'a': prior(0.)}, values)
    return graph


def test_conditional_noise_uses_pre_observation_distance_not_measurement():
    robot = IncrementalPoint2Robot(1)
    robot.push_batch([root_packet(), observation(0, distance=2.)])
    seen = []

    def noise(distance):
        seen.append(distance)
        return np.array([.0025, .01 + .001 * distance**2])

    robot.observation_variance_model = noise
    robot.push_batch([
        dict(robot=1, kind='pose', timestamp_s=2., available_time_s=2.,
             support_max_time_s=0., pose_index=1, from_pose_index=0,
             delta=[1., 0., 0.], duration_s=2.),
        observation(1, pose=1, t=2., distance=9.),
    ])
    np.testing.assert_allclose(seen, [1.], atol=1e-10)
    factor = robot.graph.factors[('observation', 1)]
    np.testing.assert_allclose(factor.noiseModel().sigmas()**2, [.0025, .011], atol=1e-10)


def test_first_observation_keeps_nominal_noise_without_independent_distance():
    robot = IncrementalPoint2Robot(1)
    robot.observation_variance_model = lambda distance: np.array([1., 1.])
    robot.push_batch([root_packet(), observation(0)])
    factor = robot.graph.factors[('observation', 0)]
    np.testing.assert_allclose(factor.noiseModel().sigmas()**2, [.0025, .01])


def test_deleted_group_calibration_rows_keep_factor_types_separate():
    innovations = [np.array([2., 0.]), np.array([-2., 1.]),
                   np.array([1., -2.])]
    deleted = [FactorDeletedBlockEvidence(
        row_offset=2 * index, dimension=2, innovation=innovation,
        predictive_covariance=np.eye(2),
        normalized_energy=float(innovation @ innovation),
        energy_per_dimension=float(innovation @ innovation / 2.),
        identifiable=True, maximum_leverage=0.,
        minimum_deleted_precision=1.)
        for index, innovation in enumerate(innovations)]
    factor_items = [(('observation', index), None)
                    for index in range(len(deleted))]

    rows = deleted_group_calibration_rows(
        factor_items, deleted, dataset=1, robot=2, elapsed_s=30.)

    assert len(rows) == 1
    assert rows[0]['factor_kind'] == 'observation'
    assert rows[0]['factor_count'] == 3
    assert rows[0]['inclusive_scale'] > 0.


def test_deleted_observation_model_recovers_range_dependence():
    rng = np.random.default_rng(25)
    factor_items, deleted = [], []
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.05, .1]))
    for index, distance in enumerate(np.linspace(.5, 6., 300)):
        bearing_error = .02 * rng.normal()
        range_error = np.sqrt(.0025 + .002 * distance**2) * rng.normal()
        factor = gtsam.BearingRangeFactor2D(
            gtsam.symbol('x', index), gtsam.symbol('l', index),
            gtsam.Rot2(), float(distance), noise)
        innovation = np.array([bearing_error / .05, range_error / .1])
        factor_items.append((('observation', index), factor))
        deleted.append(FactorDeletedBlockEvidence(
            row_offset=2 * index, dimension=2, innovation=innovation,
            predictive_covariance=np.eye(2),
            normalized_energy=float(innovation @ innovation),
            energy_per_dimension=float(innovation @ innovation / 2.),
            identifiable=True, maximum_leverage=0.,
            minimum_deleted_precision=1.))

    rows = deleted_observation_model_rows(
        factor_items, deleted, dataset=1, robot=2, elapsed_s=30.)

    assert len(rows) == 1
    assert rows[0]['range_distance_squared_coefficient'] > 0.
    assert rows[0]['range_conditional_nll_gain'] > 0.
    assert rows[0]['bearing_variance'] > 0.
    assert rows[0]['median_range_state_std'] == pytest.approx(0.)
    assert rows[0]['median_bearing_state_std'] == pytest.approx(0.)


def test_append_preserves_solver_and_old_factors():
    graph = initialized()
    identity = id(graph.solver)
    graph.update({'b': prior(2.)})
    assert id(graph.solver) == identity
    assert graph.factor_count == 2
    np.testing.assert_allclose(graph.estimate().atPoint2(1), [1., 0.], atol=1e-10)


def test_replace_noise_invalidates_old_linearized_factor():
    graph = initialized()
    graph.update({'b': prior(1.)})
    graph.update({'a': prior(0., .1)})
    assert graph.factor_count == 2
    np.testing.assert_allclose(graph.estimate().atPoint2(1), [1./101., 0.], atol=1e-10)
    rebuilt = gtsam.LevenbergMarquardtOptimizer(graph.factor_graph(), graph.estimate()).optimize()
    np.testing.assert_allclose(rebuilt.atPoint2(1), graph.estimate().atPoint2(1), atol=1e-10)


def test_remove_factor_matches_remaining_batch_objective():
    graph = initialized()
    graph.update({'b': prior(2.)})
    graph.update({}, remove=('a',))
    assert graph.factor_count == 1
    np.testing.assert_allclose(graph.estimate().atPoint2(1), [2., 0.], atol=1e-10)


def test_invalid_transaction_does_not_change_state():
    graph = initialized()
    before = graph.estimate().atPoint2(1)
    with pytest.raises(ValueError):
        graph.update({'b': prior(10.)}, remove=('missing',))
    assert graph.factor_count == 1
    np.testing.assert_array_equal(before, graph.estimate().atPoint2(1))


def test_missing_variable_rejected_before_updating_solver():
    graph = initialized()
    with pytest.raises(ValueError):
        graph.update({'bad': gtsam.PriorFactorPoint2(99, np.zeros(2), gtsam.noiseModel.Unit.Create(2))})
    graph.update({'b': prior(2.)})
    assert graph.factor_count == 2


def test_missing_ordering_constraint_is_rejected_before_solver_update():
    graph = initialized()
    with pytest.raises(ValueError):
        graph.update({'b': prior(2.)}, constrained_keys={99: 1})
    graph.update({'b': prior(2.)})
    assert graph.factor_count == 2


def test_incremental_thresholds_are_validated():
    with pytest.raises(ValueError):
        IncrementalFactorGraph(wildfire_threshold=-1.)
    with pytest.raises(ValueError):
        IncrementalFactorGraph(wildfire_threshold=float('nan'))


def test_new_landmark_ordering_is_applied_only_at_variable_birth():
    old = {gtsam.symbol('l', 6), gtsam.symbol('l', 7)}
    new = gtsam.symbol('l', 8)

    assert landmark_ordering_constraints('natural', old, new) is None
    assert landmark_ordering_constraints('new_last', old, new) == {new: 1}
    assert landmark_ordering_constraints('new_last', old, None) is None
    assert landmark_ordering_constraints('all_last', old, new) == {
        gtsam.symbol('l', 6): 1,
        gtsam.symbol('l', 7): 1,
        new: 1,
    }


def test_unknown_landmark_ordering_is_rejected():
    with pytest.raises(ValueError):
        landmark_ordering_constraints('tuned_magic', set(), None)


def root_packet():
    return dict(robot=1, kind='pose', timestamp_s=0., available_time_s=0.,
                support_max_time_s=0., pose_index=0, from_pose_index=None,
                delta=None, duration_s=0.)


def observation(row, pose=0, t=0., distance=2., bearing=0.):
    return dict(robot=1, kind='observation', timestamp_s=t, available_time_s=t,
                support_max_time_s=t, pose_index=pose, source_row=row,
                landmark_subject=6, range_m=distance, bearing_rad=bearing)


def test_raw_polar_birth_and_odometry_are_incremental():
    robot = IncrementalPoint2Robot(1)
    robot.push(root_packet())
    robot.push(observation(0))
    robot.push(dict(robot=1, kind='pose', timestamp_s=1., available_time_s=1.,
                    support_max_time_s=0., pose_index=1, from_pose_index=0,
                    delta=[1., 0., 0.], duration_s=1.))
    robot.push(observation(1, pose=1, t=1., distance=1.))
    values = robot.graph.estimate()
    np.testing.assert_allclose(values.atPoint2(gtsam.symbol('l', 6)), [2., 0.], atol=1e-10)
    assert values.atPose2(gtsam.symbol('x', 1)).x() == pytest.approx(1.)
    assert robot.graph.factor_count == 4
    assert robot.physical_graph().size() == 3
    assert [key for key, _ in robot.physical_factor_items()] == [
        ('observation', 0), ('odometry', 1), ('observation', 1)]
    assert robot.landmark_keys == {gtsam.symbol('l', 6)}


def test_new_last_ordering_preserves_the_same_incremental_state():
    natural = IncrementalPoint2Robot(1, landmark_ordering='natural')
    structured = IncrementalPoint2Robot(1, landmark_ordering='new_last')
    packets = [
        root_packet(),
        observation(0),
        dict(robot=1, kind='pose', timestamp_s=1., available_time_s=1.,
             support_max_time_s=0., pose_index=1, from_pose_index=0,
             delta=[1., 0., 0.], duration_s=1.),
        observation(1, pose=1, t=1., distance=1.),
    ]
    for packet in packets:
        natural.push(packet)
        structured.push(packet)

    np.testing.assert_allclose(
        structured.graph.estimate().atPoint2(gtsam.symbol('l', 6)),
        natural.graph.estimate().atPoint2(gtsam.symbol('l', 6)), atol=1e-10)
    assert structured.graph.estimate().atPose2(
        gtsam.symbol('x', 1)).equals(
            natural.graph.estimate().atPose2(gtsam.symbol('x', 1)), 1e-10)


def test_atomic_sensor_frames_match_sequential_factor_insertion():
    sequential = IncrementalPoint2Robot(1)
    atomic = IncrementalPoint2Robot(1)
    frames = [
        [root_packet(), observation(0)],
        [dict(robot=1, kind='pose', timestamp_s=1., available_time_s=1.,
              support_max_time_s=0., pose_index=1, from_pose_index=0,
              delta=[1., 0., 0.], duration_s=1.),
         observation(1, pose=1, t=1., distance=1.)],
    ]
    for frame in frames:
        for packet in frame:
            sequential.push(packet)
        atomic.push_batch(frame)

    assert atomic.graph.factor_count == sequential.graph.factor_count
    np.testing.assert_allclose(
        atomic.graph.estimate().atPoint2(gtsam.symbol('l', 6)),
        sequential.graph.estimate().atPoint2(gtsam.symbol('l', 6)), atol=1e-10)
    assert atomic.graph.estimate().atPose2(gtsam.symbol('x', 1)).equals(
        sequential.graph.estimate().atPose2(gtsam.symbol('x', 1)), 1e-10)


def test_atomic_sensor_frame_rejects_mixed_timestamps():
    robot = IncrementalPoint2Robot(1)
    with pytest.raises(ValueError):
        robot.push_batch([root_packet(), observation(0, t=1.)])


def test_future_support_and_duplicate_measurement_are_rejected():
    robot = IncrementalPoint2Robot(1)
    robot.push(root_packet())
    packet = observation(0)
    packet['support_max_time_s'] = 1.
    with pytest.raises(ValueError):
        robot.push(packet)
    robot.push(observation(0))
    with pytest.raises(ValueError):
        robot.push(observation(0))
    assert robot.graph.factor_count == 2


def test_no_silent_cross_robot_or_nonmonotone_pose_insertion():
    robot = IncrementalPoint2Robot(1)
    robot.push(root_packet())
    packet = observation(0)
    packet['robot'] = 2
    with pytest.raises(ValueError):
        robot.push(packet)
    packet = root_packet()
    packet['pose_index'] = 2
    with pytest.raises(ValueError):
        robot.push(packet)


def test_physical_linearization_ignores_values_absent_from_factors():
    graph = gtsam.NonlinearFactorGraph()
    graph.push_back(prior(0.))
    values = gtsam.Values()
    values.insert(1, np.zeros(2))
    values.insert(2, np.ones(2))
    a, residual, keys = linearize(graph, values)
    assert keys == [1]
    assert a.shape == (2, 2)
    np.testing.assert_allclose(residual, 0.)


def test_zero_duration_replays_full_stream_without_event_csv(tmp_path):
    stream_dir = tmp_path / 'dataset1'
    output = tmp_path / 'output'
    stream_dir.mkdir()
    output.mkdir()
    packets = [
        root_packet(),
        observation(0),
        dict(robot=1, kind='pose', timestamp_s=2., available_time_s=2.,
             support_max_time_s=0., pose_index=1, from_pose_index=0,
             delta=[1., 0., 0.], duration_s=2.),
        observation(1, pose=1, t=2., distance=1.),
    ]
    event_path = stream_dir / 'events.jsonl'
    event_path.write_text(''.join(json.dumps(packet) + '\n' for packet in packets))
    digest = hashlib.sha256(event_path.read_bytes()).hexdigest()
    (stream_dir / 'metadata.json').write_text(json.dumps(dict(
        dataset=1, first_event_time_s=0., event_sha256=digest)))

    summary = native_audit(stream_dir, duration=0., checkpoints=[],
                           output=output, write_updates=False)

    assert summary['updates'] == 4
    assert summary['observation_updates'] == 2
    assert summary['duration_s'] == pytest.approx(2.)
    assert summary['process_peak_rss_mb'] > 0.
    assert summary['calibration_rows'] == 0
    assert summary['observation_model_rows'] == 0
    assert summary['profiled_calibration_rows'] == 0
    assert not (output / 'dataset1_updates.csv').exists()
    assert (output / 'dataset1_final_estimates.csv').exists()
    rows = list(csv.DictReader((output / 'dataset1_final_estimates.csv').open()))
    assert sum(row['kind'] == 'pose' for row in rows) == 2
    assert sum(row['kind'] == 'landmark' for row in rows) == 1


def test_atomic_frame_replay_counts_packets_and_solver_transactions(tmp_path):
    stream_dir = tmp_path / 'dataset1'
    output = tmp_path / 'output'
    stream_dir.mkdir()
    output.mkdir()
    packets = [
        root_packet(),
        observation(0),
        dict(robot=1, kind='pose', timestamp_s=2., available_time_s=2.,
             support_max_time_s=0., pose_index=1, from_pose_index=0,
             delta=[1., 0., 0.], duration_s=2.),
        observation(1, pose=1, t=2., distance=1.),
    ]
    event_path = stream_dir / 'events.jsonl'
    event_path.write_text(''.join(json.dumps(packet) + '\n' for packet in packets))
    digest = hashlib.sha256(event_path.read_bytes()).hexdigest()
    (stream_dir / 'metadata.json').write_text(json.dumps(dict(
        dataset=1, first_event_time_s=0., event_sha256=digest)))

    summary = native_audit(stream_dir, duration=0., checkpoints=[],
                           output=output, write_updates=False,
                           atomic_frames=True)

    assert summary['packets'] == 4
    assert summary['updates'] == 2
    assert summary['observation_updates'] == 2
    assert summary['atomic_frames']
