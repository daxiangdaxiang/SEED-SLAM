import pytest

from experiments.analyze_full_lazy_refresh import recovery_diagnostics, temporal_profile


def trace():
    return [dict(elapsed_s=str(i), accepted=str(accepted), refresh_requests=str(request),
                 refresh_rounds=str(refresh), total_ms=str(10+i), serial_backlog_s='.1',
                 optimizer_bytes='100', control_bytes='200')
            for i, (accepted, request, refresh) in enumerate([
                (1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (0, 1, 0)])]


def test_recovery_retains_consecutive_failures_and_terminal_pending_event():
    result = recovery_diagnostics(trace())
    assert result['stalled_frames'] == 3
    assert result['longest_stalled_run_frames'] == 2
    assert result['longest_repeated_refresh_requests'] == 2
    assert result['next_frame_full_step_recoveries'] == 1
    assert result['requests_without_next_frame'] == 1
    assert result['request_events'][-1]['next_accepted'] is None


def test_temporal_profiles_do_not_realign_or_hide_unsupported_poses():
    result = temporal_profile(trace(), [1., 4., None, 16., 25.])
    assert len(result) == 4
    assert sum(r['frames'] for r in result) == 5
    assert sum(r['supported_poses'] for r in result) == 4
    assert result[0]['measurement_rmse_m'] == 1.
    assert result[2]['measurement_rmse_m'] is None
    assert result[3]['measurement_rmse_m'] == pytest.approx((41./2)**.5)
    assert sum(r['application_bytes'] for r in result) == 1500
    assert result[3]['service_seconds'] == pytest.approx(.027)


def test_temporal_profile_rejects_misaligned_or_nonfinite_error_vectors():
    with pytest.raises(ValueError, match='length'):
        temporal_profile(trace(), [0.])
    with pytest.raises(ValueError, match='squared error'):
        temporal_profile(trace(), [0., 0., float('nan'), 0., 0.])
