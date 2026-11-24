from pathlib import Path
import sys

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import audit_online_lower
from audit_online_noise import predicted_observations
from test_audit_online_noise import initial_frame, next_frame


def test_checkpoint_excludes_whole_target_frame_and_all_future_data():
    import audit_online_checkpoint as checkpoint
    packets = initial_frame() + next_frame(1, 1.9) + next_frame(2, 1.8)
    prefix, frame, elapsed = checkpoint.prefix_before_source(packets, 1, 2)
    assert len(prefix) == 4
    assert frame == tuple(next_frame(2, 1.8))
    assert elapsed == 2.
    assert all(row['timestamp_s'] < frame[0]['timestamp_s'] for row in prefix)
    changed = initial_frame() + next_frame(1, 1.9) + next_frame(2, 100.)
    assert checkpoint.prefix_before_source(changed, 1, 2)[0] == prefix
    with pytest.raises(ValueError, match='source row'):
        checkpoint.prefix_before_source(packets, 1, 99)


def test_prediction_at_refined_values_does_not_mutate_online_graph():
    import audit_online_checkpoint as checkpoint
    robots, _ = audit_online_lower.replay_fixed_calibration(
        initial_frame() + next_frame(1, 1.9), [], .5, 1., methods=('fixed',))
    robot = robots['fixed'][1]
    before = robot.graph.estimate()
    alternate = gtsam.Values(before)
    key = gtsam.symbol('l', 6)
    alternate.erase(key)
    alternate.insertPoint2(key, before.atPoint2(key) + np.array([.4, .2]))
    nominal = predicted_observations(robot, next_frame(2))[0]
    refined = checkpoint.prediction_at_values(robot, next_frame(2), alternate)[0]
    assert refined['predicted_range'] != pytest.approx(nominal['predicted_range'])
    np.testing.assert_array_equal(robot.graph.estimate().atPoint2(key), before.atPoint2(key))
    assert robot.pose_index == 1
