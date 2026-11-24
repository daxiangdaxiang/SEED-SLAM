import pytest

from experiments import analyze_lazy_stress as analysis


def rows():
    return [dict(robot='1', timestamp_s=str(t), accepted=str(accepted),
                 refresh_requests=str(request), full_step_prediction='1',
                 full_step_decrease=str(decrease))
            for t, accepted, request, decrease in [(1., 1, 0, 1.), (2., 1, 1, -1.), (3., 0, 1, -2.)]]


def test_stress_status_counts_frames_not_observations():
    changes = [dict(robot=1, timestamp_s=1.)]*2+[dict(robot=1, timestamp_s=3.)]
    result = analysis.frame_status(rows(), changes)
    assert result['injected'] == dict(frames=2, accepted=1, requests=1, full_step_supported=1)
    assert result['not_injected'] == dict(frames=1, accepted=1, requests=1, full_step_supported=0)


def test_stress_status_rejects_mismatched_frame_provenance():
    with pytest.raises(ValueError, match='missing'):
        analysis.frame_status(rows(), [dict(robot=2, timestamp_s=1.)])
