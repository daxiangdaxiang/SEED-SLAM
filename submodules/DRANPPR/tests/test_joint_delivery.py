import numpy as np
import pytest

from benchmarks.mrclam.evaluate import GroundTruthData,RobotGroundTruthSeries
from experiments.analyze_joint_delivery import delivery_metrics


def test_exact_timestamped_state_can_be_stale_at_delivery():
    truth=GroundTruthData(robots={1:RobotGroundTruthSeries(1,np.arange(5.),
        np.column_stack((np.arange(5.),np.zeros(5))),np.zeros(5))},landmarks={})
    rows=[dict(robot=1,timestamp_s=float(i),x=float(i),y=0.) for i in [0,1,2]]
    updates=[dict(robot=1,timestamp_s=float(i),serial_backlog_s=.5) for i in [0,1,2]]
    alignment=dict(alignment_x=0.,alignment_y=0.,alignment_yaw=0.)
    result=delivery_metrics(rows,updates,truth,alignment)
    assert result['measurement_time_rmse_m'] == 0.
    assert result['delivery_time_stale_pose_rmse_m'] == pytest.approx(.5)
    assert result['state_age_p99_s'] == .5
    updates[0]['serial_backlog_s']=-1.
    with pytest.raises(ValueError,match='invalid service age'):
        delivery_metrics(rows,updates,truth,alignment)
