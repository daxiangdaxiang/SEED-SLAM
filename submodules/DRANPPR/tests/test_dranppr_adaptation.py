import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_adaptation import (
    UnobservablePredictionError,
    candidate_predictive_nll,
    gnc_geman_mcclure_fit,
    huber_state_fit,
    naive_scale_from_state,
    predictive_group_scale,
    weighted_state_fit,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from audit_cause_separated_adaptation import audit, summarize


def problem(seed=0, sensor_scale=1.):
    rng = np.random.default_rng(seed)
    dimension = 5
    truth = rng.normal(size=dimension)
    reference_a = rng.normal(size=(25, dimension))
    sensor_a = rng.normal(size=(50, dimension))
    reference_y = reference_a @ truth + rng.normal(size=25)
    sensor_y = sensor_a @ truth + np.sqrt(sensor_scale) * rng.normal(size=50)
    return truth, reference_a, reference_y, sensor_a, sensor_y


def test_predictive_scale_recovers_group_noise_without_initial_state():
    _, reference_a, reference_y, sensor_a, sensor_y = problem(
        seed=2, sensor_scale=9.)

    estimate = predictive_group_scale(
        reference_a, reference_y, sensor_a, sensor_y)

    assert estimate == pytest.approx(9., abs=3.)


def test_naive_scale_absorbs_state_error_but_predictive_scale_does_not():
    truth, reference_a, reference_y, sensor_a, sensor_y = problem(seed=4)
    bad_state = truth + 10.

    naive = naive_scale_from_state(sensor_a, sensor_y, bad_state)
    predictive = predictive_group_scale(
        reference_a, reference_y, sensor_a, sensor_y)

    assert naive > 100.
    assert predictive == pytest.approx(1., abs=.6)


def test_candidate_likelihood_prefers_the_independently_supported_relation():
    truth, reference_a, reference_y, peers_a, peers_y = problem(seed=8)
    rng = np.random.default_rng(9)
    true_row = rng.normal(size=len(truth))
    wrong_row = rng.normal(size=len(truth))
    measurement = float(true_row @ truth + .1)
    scale = predictive_group_scale(reference_a, reference_y, peers_a, peers_y)

    scores = candidate_predictive_nll(
        reference_a, reference_y, peers_a, peers_y,
        np.vstack([true_row, wrong_row]), measurement, scale)

    assert scores[0] < scores[1]


def test_weighted_state_fit_rejects_invalid_variance():
    with pytest.raises(ValueError):
        weighted_state_fit(np.eye(2), np.ones(2), np.array([1., 0.]))


def test_prediction_cannot_rank_an_unobserved_direction_as_zero_variance():
    reference = np.array([[1., 0.], [2., 0.]])
    with pytest.raises(UnobservablePredictionError, match='observable'):
        candidate_predictive_nll(reference, np.array([1., 2.]),
            np.empty((0, 2)), np.empty(0), np.eye(2), 0., 1.)


def test_observable_relative_prediction_does_not_require_absolute_gauge():
    design = np.array([[1., -1.], [2., -2.]])
    score = candidate_predictive_nll(design, np.array([2., 4.]),
        np.empty((0, 2)), np.empty(0), np.array([[1., -1.]]), 2., 1.)
    assert score[0] == pytest.approx(.5*np.log(2.*np.pi*1.2))


def test_weak_observable_direction_retains_its_large_uncertainty():
    design = np.diag([1., 1e-8])
    measurement = np.array([0., 1e-8])
    fit = weighted_state_fit(design, measurement, np.ones(2))
    assert fit.rank == 2
    np.testing.assert_allclose(fit.covariance, np.diag([1., 1e16]), rtol=1e-12)
    score = candidate_predictive_nll(design, measurement,
        np.empty((0, 2)), np.empty(0), np.array([[0., 1.]]), 1., 1.)
    assert score[0] == pytest.approx(.5*np.log(2.*np.pi*(1.+1e16)))


def test_prediction_variance_avoids_cancellation_of_a_weak_shared_mode():
    fit = weighted_state_fit(np.array([[1., 1.], [1e-8, -1e-8]]),
        np.array([2., 0.]), np.ones(2))
    means, variances = fit.predict(np.array([[1., 1.], [1e-8, -1e-8]]))
    np.testing.assert_allclose(means, [2., 0.], atol=1e-12)
    np.testing.assert_allclose(variances, [1., 1.], rtol=1e-12)


def test_scale_fit_requires_support_after_each_factor_exclusion():
    with pytest.raises(UnobservablePredictionError, match='observable'):
        predictive_group_scale(np.array([[1., 0.], [2., 0.]]),
            np.array([1., 2.]), np.array([[0., 1.], [1., 0.]]), np.array([.5, 1.]))


def test_prediction_is_invariant_to_invertible_state_coordinates():
    design = np.array([[1., -1., 0.], [0., 1., -1.]])
    measurement, variance = np.array([1., 2.]), np.array([.3, .7])
    rows = np.array([[1., 0., -1.], [0., 0., 0.]])
    change = np.array([[2., 1., .1], [.3, 1., .2], [0., .2, 3.]])
    a = weighted_state_fit(design, measurement, variance)
    b = weighted_state_fit(design @ change, measurement, variance)
    assert a.rank == b.rank == 2
    means_a, variances_a = a.predict(rows)
    means_b, variances_b = b.predict(rows @ change)
    np.testing.assert_allclose(means_a, [3., 0.], atol=1e-12)
    np.testing.assert_allclose(variances_a, [1., 0.], atol=1e-12)
    np.testing.assert_allclose(means_b, means_a, atol=1e-12)
    np.testing.assert_allclose(variances_b, variances_a, atol=1e-12)
    with pytest.raises(UnobservablePredictionError):
        b.predict(np.array([[1., 0., 0.]]) @ change)


def test_zero_information_only_supports_state_independent_predictions():
    fit = weighted_state_fit(np.zeros((3, 2)), np.arange(3.), np.ones(3))
    assert fit.rank == 0
    mean, variance = fit.predict(np.zeros((1, 2)))
    np.testing.assert_array_equal(mean, [0.])
    np.testing.assert_array_equal(variance, [0.])
    with pytest.raises(UnobservablePredictionError):
        fit.predict(np.array([[1e-20, 0.]]))
    with pytest.raises(ValueError, match='invalid prediction'):
        fit.predict(np.array([[np.nan, 0.]]))


def test_full_rank_state_and_covariance_match_independent_direct_solve():
    rng = np.random.default_rng(271)
    design, measurement = rng.normal(size=(12, 4)), rng.normal(size=12)
    variance = np.exp(rng.normal(size=12))
    fit = weighted_state_fit(design, measurement, variance)
    whitened = design/np.sqrt(variance[:, None])
    expected, _, _, _ = np.linalg.lstsq(whitened, measurement/np.sqrt(variance), rcond=None)
    np.testing.assert_allclose(fit.state, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.covariance,
        np.linalg.solve(whitened.T @ whitened, np.eye(4)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('fit', [huber_state_fit, gnc_geman_mcclure_fit])
def test_robust_control_matches_exact_clean_linear_solution(fit):
    design = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
    truth = np.array([2., -3.])

    result = fit(design, design @ truth, np.ones(len(design)))

    np.testing.assert_allclose(result.state, truth, atol=1e-10)
    np.testing.assert_allclose(result.weights, 1., atol=1e-10)


@pytest.mark.parametrize('fit', [huber_state_fit, gnc_geman_mcclure_fit])
def test_robust_control_reduces_one_gross_outlier_influence(fit):
    design = np.ones((11, 1))
    measurement = np.concatenate([np.zeros(10), [100.]])
    least_squares = weighted_state_fit(
        design, measurement, np.ones(len(design))).state[0]

    result = fit(design, measurement, np.ones(len(design)))

    assert abs(result.state[0]) < abs(least_squares)
    assert result.weights[-1] < result.weights[0]


def test_controlled_audit_keeps_state_sensor_and_relation_arms_separate():
    summary = summarize(audit(4))

    assert summary['seeds'] == 4
    assert summary['nominal']['naive_scale_inflation'] > 10.
    assert summary['nominal']['fixed_to_oracle_rmse'] == pytest.approx(1.)
    assert summary['nominal']['huber_to_oracle_rmse'] > 0.
    assert summary['nominal']['gnc_gm_to_oracle_rmse'] > 0.
    assert summary['sensor_change']['predictive_scale_ratio'] > 3.
    assert summary['sensor_change']['naive_scale_ratio'] > 10.
    assert summary['sensor_change']['naive_to_oracle_rmse'] > 1.
    assert summary['sensor_change']['huber_sensor_weight'] < 1.
    assert summary['sensor_change']['gnc_gm_sensor_weight'] < 1.
    assert summary['wrong_relation']['true_candidate_fraction'] == 1.
    assert summary['wrong_relation']['inflated_wrong_to_oracle_rmse'] > 1.
    assert summary['wrong_relation']['huber_target_weight'] < 1.
    assert summary['wrong_relation']['gnc_gm_target_weight'] < 1.
    assert summary['wrong_relation']['equivalent_candidate_nll_difference'] < 1e-12
