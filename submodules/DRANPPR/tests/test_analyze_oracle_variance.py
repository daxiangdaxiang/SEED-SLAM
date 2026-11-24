import math

import pytest

from experiments import analyze_oracle_variance as analysis


def test_common_alignment_applies_rotation_then_translation():
    alignment = dict(alignment_yaw=math.pi/2, alignment_x=3., alignment_y=-2.)
    row = dict(x='2', y='1')
    assert analysis.point_error_squared(row, [2., 0.], alignment) == pytest.approx(0., abs=1e-25)
    assert analysis.point_error_squared(row, [5., 4.], alignment) == pytest.approx(25.)


def test_common_alignment_metric_rejects_invalid_inputs():
    alignment = dict(alignment_yaw=0., alignment_x=0., alignment_y=0.)
    with pytest.raises(ValueError, match='nonfinite'):
        analysis.point_error_squared(dict(x='nan', y='0'), [0., 0.], alignment)
