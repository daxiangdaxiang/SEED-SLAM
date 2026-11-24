import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from evaluate_incremental_reference import evaluate_rows
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.mrclam.evaluate import GroundTruthData, RobotGroundTruthSeries


def test_each_local_robot_is_aligned_before_state_evaluation():
    truth = GroundTruthData(
        robots={1: RobotGroundTruthSeries(
            1, np.array([0., 1., 2.]),
            np.array([[0., 0.], [1., 0.], [2., 0.]]),
            np.zeros(3))},
        landmarks={6: np.array([1., 2.])})
    # Local-from-world is a +90 degree rotation and translation. The estimated
    # trajectory and object are exact in that arbitrary local gauge.
    world = np.array([[0., 0.], [1., 0.], [2., 0.], [1., 2.]])
    rotation = np.array([[0., 1.], [-1., 0.]])
    local = (rotation @ world.T).T + np.array([3., -4.])
    rows = [dict(robot='1', kind='pose', index=str(i), timestamp_s=str(i),
                 x=str(local[i, 0]), y=str(local[i, 1]), yaw=str(-np.pi/2))
            for i in range(3)]
    rows.append(dict(robot='1', kind='landmark', index='6', timestamp_s='',
                     x=str(local[3, 0]), y=str(local[3, 1]), yaw=''))

    aggregate, robots = evaluate_rows(rows, truth, {})

    assert aggregate['translation_rmse_m'] == pytest.approx(0., abs=1e-12)
    assert aggregate['yaw_rmse_rad'] == pytest.approx(0., abs=1e-12)
    assert aggregate['object_copy_rmse_m'] == pytest.approx(0., abs=1e-12)
    assert robots[0]['pose_count'] == 3


def test_poses_outside_ground_truth_support_are_counted_and_excluded():
    truth = GroundTruthData(
        robots={1: RobotGroundTruthSeries(
            1, np.array([0., 1., 2.]),
            np.array([[0., 0.], [1., 0.], [2., 0.]]),
            np.zeros(3))},
        landmarks={})
    rows = [dict(robot='1', kind='pose', index=str(i), timestamp_s=str(t),
                 x=str(t), y='0', yaw='0')
            for i, t in enumerate([0., 1., 2., 3.])]

    aggregate, robots = evaluate_rows(rows, truth, {})

    assert aggregate['pose_count'] == 3
    assert aggregate['unsupported_pose_count'] == 1
    assert robots[0]['unsupported_pose_count'] == 1
    assert aggregate['translation_rmse_m'] == pytest.approx(0., abs=1e-12)
