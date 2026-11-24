from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))


def test_factorial_arms_use_identical_fit_then_change_only_declared_covariance():
    from audit_noise_factorial import ablate_calibration
    fit = dict(bearing_variance=.001, range_intercept_variance=.002,
               range_distance_squared_coefficient=.003,
               odometry_variance_scales=[.5, .02, 1.1], success=True)
    q = ablate_calibration(fit, 'q_only')
    r = ablate_calibration(fit, 'r_only')
    joint = ablate_calibration(fit, 'reml_diagonal_q')
    assert q['odometry_variance_scales'] == fit['odometry_variance_scales']
    assert q['bearing_variance'] == pytest.approx(.05**2)
    assert q['range_intercept_variance'] == pytest.approx(.1**2)
    assert q['range_distance_squared_coefficient'] == 0.
    assert r['odometry_variance_scales'] == [1., 1., 1.]
    for key in ('bearing_variance', 'range_intercept_variance', 'range_distance_squared_coefficient'):
        assert r[key] == fit[key] == joint[key]
    assert q['unablated_fit'] == r['unablated_fit'] == joint['unablated_fit'] == fit
    assert 'method' not in fit


def test_dense_reference_comparison_checks_complete_matching_event_sets():
    from audit_noise_factorial import compare_reference
    rows = [dict(method='fixed', robot=3, source_row=4, nll=1., energy=2.,
                 bearing_innovation=.1, range_innovation=.2,
                 state_bearing_variance=.03, state_range_variance=.04,
                 state_cross_covariance=.001)]
    assert all(row['passed'] for row in compare_reference(rows, rows, 'reference'))
    with pytest.raises(ValueError, match='event'):
        compare_reference(rows, [dict(rows[0], source_row=9)], 'reference')
    bad = [dict(rows[0], state_cross_covariance=.05)]
    assert not all(row['passed'] for row in compare_reference(rows, bad, 'reference'))
