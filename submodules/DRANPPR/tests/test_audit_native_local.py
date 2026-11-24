import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from audit_native_local import state_difference


def test_state_difference_does_not_align_or_mix_angle_and_distance():
    reference = [dict(robot='1', kind='pose', index='0', x='0', y='0', yaw='3.1')]
    candidate = [dict(reference[0], x='3', y='4', yaw=str(3.1+2*3.141592653589793))]
    difference = state_difference(candidate, reference)
    assert difference['max_translation_to_exact_lm'] == 5.
    assert difference['max_yaw_to_exact_lm'] < 1e-12
    with pytest.raises(ValueError, match='keys'):
        state_difference([], reference)
