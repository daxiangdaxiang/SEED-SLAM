import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_profiled_noise import DeletedLinearLikelihood, fit_polar_noise


def test_deleted_likelihood_matches_explicit_weighted_refits():
    rng = np.random.default_rng(53)
    a = rng.normal(size=(24, 5))
    y = rng.normal(size=24)
    variance = np.exp(rng.normal(size=24))
    model = DeletedLinearLikelihood(a, y, [2] * 12, list(range(12)))
    expected = 0.
    for i in range(12):
        rows = np.arange(2 * i, 2 * i + 2)
        keep = np.ones(24, dtype=bool)
        keep[rows] = False
        train = a[keep] / np.sqrt(variance[keep, None])
        state = np.linalg.lstsq(train, y[keep] / np.sqrt(variance[keep]), rcond=None)[0]
        covariance = np.diag(variance[rows]) + a[rows] @ np.linalg.inv(
            train.T @ train) @ a[rows].T
        innovation = y[rows] - a[rows] @ state
        expected += .5 * (np.linalg.slogdet(covariance)[1] +
                          innovation @ np.linalg.solve(covariance, innovation) +
                          2 * np.log(2 * np.pi))
    assert model.nll(variance) == pytest.approx(expected, rel=1e-10)


def test_deleted_likelihood_preserves_gauge_and_state_shift_invariance():
    rng = np.random.default_rng(54)
    a = rng.normal(size=(20, 4))
    a = np.column_stack([a, a[:, 0] + a[:, 1]])
    y = rng.normal(size=20)
    v = np.exp(rng.normal(size=20))
    plain = DeletedLinearLikelihood(a, y, [2] * 10, list(range(10)))
    shifted = DeletedLinearLikelihood(a, y + a @ np.ones(5), [2] * 10, list(range(10)))
    assert plain.rank == 4
    assert shifted.nll(v) == pytest.approx(plain.nll(v), rel=1e-10)


def test_unobservable_blocks_are_reported_not_scored():
    a = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    model = DeletedLinearLikelihood(a, np.array([0., 2., 1., -1.]),
                                    [1] * 4, [1, 2, 3])
    assert model.scored_blocks == (2, 3)
    assert model.unobservable_blocks == (1,)
    assert np.isfinite(model.nll(np.ones(4)))


def test_profiled_conditional_noise_recovers_sensor_parameters():
    rng = np.random.default_rng(55)
    count = 500
    reference = rng.normal(size=(30, 4))
    sensor = rng.normal(size=(2 * count, 4))
    distances = rng.uniform(.5, 7., count)
    variance = np.ones(30 + 2 * count)
    variance[30::2] = .4
    variance[31::2] = .3 + .2 * distances**2
    a = np.vstack([reference, sensor])
    y = a @ rng.normal(size=4) + np.sqrt(variance) * rng.normal(size=len(a))
    fit = fit_polar_noise(a, y, [1] * 30 + [2] * count,
                         list(range(30, 30 + count)), distances,
                         bearing_sigma=1., range_sigma=1., conditional=True)
    assert fit['success']
    assert fit['bearing_variance'] == pytest.approx(.4, abs=.08)
    assert fit['range_intercept_variance'] == pytest.approx(.3, abs=.16)
    assert fit['range_distance_squared_coefficient'] == pytest.approx(.2, abs=.04)
    assert not fit['at_numerical_boundary']


@pytest.mark.parametrize('variance', [np.zeros(4), -np.ones(4), [1., 1., np.nan, 1.]])
def test_deleted_likelihood_rejects_invalid_variance(variance):
    model = DeletedLinearLikelihood(np.ones((4, 1)), np.arange(4.), [1] * 4, list(range(4)))
    with pytest.raises(ValueError):
        model.nll(variance)
