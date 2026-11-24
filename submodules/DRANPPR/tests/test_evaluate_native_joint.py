import numpy as np
import pytest

from benchmarks.mrclam.evaluate import GroundTruthData, RobotGroundTruthSeries
from experiments.evaluate_native_joint import evaluate_joint_rows


def fixture():
    truth = GroundTruthData(robots={r: RobotGroundTruthSeries(r, np.arange(3.),
        np.array([[0., r], [1., r], [2., r]]), np.zeros(3)) for r in [1, 2]},
        landmarks={6: np.array([1., 3.])})
    rows = [dict(robot=r, kind='pose', index=k, timestamp_s=float(k), x=float(k)+7.,
                 y=float(r)-2., yaw=0.) for r in [1, 2] for k in range(3)]
    rows.extend(dict(robot=r, kind='landmark', index=6, x=8., y=1.) for r in [1, 2])
    return rows, truth


def test_one_global_alignment_removes_only_common_gauge():
    rows, truth = fixture()
    result = evaluate_joint_rows(rows, truth, {})
    assert result['translation_rmse_m'] < 1e-12
    assert result['object_copy_rmse_m'] < 1e-12
    assert result['object_copy_count'] == 2
    for row in rows:
        if row['robot'] == 2:
            row['y'] += 2.
    result = evaluate_joint_rows(rows, truth, {})
    assert result['translation_rmse_m'] == pytest.approx(1.)
    assert result['object_copy_rmse_m'] == pytest.approx(1.)


def test_unsupported_truth_timestamps_remain_explicit():
    rows, truth = fixture()
    rows.append(dict(robot=1, kind='pose', index=3, timestamp_s=9., x=99., y=99., yaw=0.))
    result = evaluate_joint_rows(rows, truth, {})
    assert result['unsupported_pose_count'] == 1
    assert result['pose_count'] == 6
    assert result['translation_rmse_m'] < 1e-12
