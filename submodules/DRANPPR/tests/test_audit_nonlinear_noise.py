from pathlib import Path
import sys

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from audit_online_lower import replay_fixed_calibration
from test_audit_online_noise import initial_frame, next_frame


def test_deleted_linear_prediction_matches_explicit_weighted_fit():
    from audit_nonlinear_noise import deleted_prediction
    rng = np.random.default_rng(27)
    a = rng.normal(size=(25, 4))
    y = rng.normal(size=25)
    sigmas = np.exp(rng.normal(size=25))
    aw, residual = a / sigmas[:, None], -y / sigmas
    held = np.array([3, 8, 19, 22])
    keep = np.setdiff1d(np.arange(25), held)
    innovation, covariance = deleted_prediction(aw, residual, held, sigmas[held], True)
    state = np.linalg.lstsq(aw[keep], -residual[keep], rcond=None)[0]
    expected = np.diag(sigmas[held]**2) + a[held] @ np.linalg.inv(aw[keep].T @ aw[keep]) @ a[held].T
    np.testing.assert_allclose(innovation, y[held] - a[held] @ state, rtol=1e-11)
    np.testing.assert_allclose(covariance, expected, rtol=1e-11)
    altered = residual.copy()
    altered[held] += 5.
    changed, changed_cov = deleted_prediction(aw, altered, held, sigmas[held], True)
    np.testing.assert_allclose(covariance, changed_cov)
    np.testing.assert_allclose(changed - innovation, -5. * sigmas[held])


def test_frame_groups_remove_observations_not_odometry():
    from audit_nonlinear_noise import observation_groups
    robots, _ = replay_fixed_calibration(initial_frame() + next_frame(1), [], .5, 1., ('fixed',))
    items = robots['fixed'][1].physical_factor_items()
    original = items[-1][1]
    items.append((('observation', 99), original))
    groups = observation_groups(items, 'frame')
    assert [len(group) for group in groups] == [1, 2]
    assert all(items[i][0][0] == 'observation' for group in groups for i in group)
    assert len(observation_groups(items, 'factor')) == 3


def test_reweighting_is_explicit_and_does_not_mutate_existing_factors():
    from audit_nonlinear_noise import reweight_items
    robots, _ = replay_fixed_calibration(initial_frame() + next_frame(1), [], .5, 1., ('fixed',))
    items = robots['fixed'][1].physical_factor_items()
    new = reweight_items(items, {0: np.array([.001, .004])})
    np.testing.assert_allclose(new[0][1].noiseModel().sigmas()**2, [.001, .004])
    np.testing.assert_allclose(items[0][1].noiseModel().sigmas()**2, [.05**2, .1**2])
    assert new[1][1] is items[1][1]


def test_unobservable_deleted_model_is_not_silently_regularized():
    from audit_nonlinear_noise import deleted_prediction
    with pytest.raises(ValueError, match='observable'):
        deleted_prediction(np.eye(3), np.ones(3), np.array([1]), np.ones(1), True)


def test_gaussian_nll_retains_cross_covariance_and_physical_units():
    from audit_nonlinear_noise import gaussian_nll
    r = np.array([.3, -.1])
    covariance = np.array([[.04, .01], [.01, .02]])
    expected = .5 * (r @ np.linalg.solve(covariance, r) +
                     np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi))
    assert gaussian_nll(r, covariance) == pytest.approx(expected)


def test_variance_components_preserve_unscored_observations():
    from audit_contrast_noise import noise_components
    robots, _ = replay_fixed_calibration(initial_frame() + next_frame(1), [], .5, 1., ('fixed',))
    items = robots['fixed'][1].physical_factor_items()
    components, base, scale = noise_components(items, {0: 2.})
    assert scale == 2.
    np.testing.assert_allclose(base, [0., 0., 0., 0., 0., 1., 1.])
    np.testing.assert_allclose(base + components @ [1., 1., .5, .5], np.ones(7))


def test_diagonal_odometry_components_follow_local_tangent_coordinates():
    from audit_contrast_noise import noise_components
    robots, _ = replay_fixed_calibration(initial_frame() + next_frame(1), [], .5, 1., ('fixed',))
    items = robots['fixed'][1].physical_factor_items()
    d, base, _ = noise_components(items, {0: 2.}, diagonal_odometry=True)
    np.testing.assert_allclose(d[2:5, :3], np.eye(3))
    np.testing.assert_allclose(base+d @ [2., 3., 4., 1., .5, .5], [1., 1., 2., 3., 4., 1., 1.])
