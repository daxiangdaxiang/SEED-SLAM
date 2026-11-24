import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_conditional_noise import (
    fit_constant_measurement_variance,
    fit_range_conditional_variance,
    gaussian_mean_nll,
)


def test_range_model_improves_heldout_prediction_without_absorbing_state_variance():
    rng = np.random.default_rng(4)
    distance = rng.uniform(.5, 8., 8000)
    state_variance = .04 + .01 * rng.random(len(distance))
    measurement_variance = .01 + .004 * distance**2
    innovation = rng.normal(scale=np.sqrt(state_variance + measurement_variance))
    train = np.arange(len(distance)) % 2 == 0
    constant = fit_constant_measurement_variance(innovation[train], state_variance[train])
    conditional = fit_range_conditional_variance(
        innovation[train], distance[train], state_variance[train])
    constant_test = gaussian_mean_nll(
        innovation[~train], state_variance[~train] + constant.variance)
    conditional_test = gaussian_mean_nll(
        innovation[~train], state_variance[~train] +
        conditional.measurement_variance(distance[~train]))

    assert constant.success and conditional.success
    assert conditional_test < constant_test - .05
    assert conditional.intercept_variance == pytest.approx(.01, abs=.006)
    assert conditional.distance_squared_coefficient == pytest.approx(.004, abs=.0006)


def test_constant_fit_subtracts_known_state_prediction_variance():
    rng = np.random.default_rng(9)
    state_variance = np.linspace(.01, .2, 10000)
    innovation = rng.normal(scale=np.sqrt(state_variance + .04))
    result = fit_constant_measurement_variance(innovation, state_variance)
    assert result.variance == pytest.approx(.04, abs=.004)


@pytest.mark.parametrize('innovation,distance,state', [
    ([1.], [-1.], [0.]),
    ([1., 2.], [1.], [0., 0.]),
    ([1.], [1.], [-1.]),
])
def test_invalid_predictive_inputs_are_rejected(innovation, distance, state):
    with pytest.raises(ValueError):
        fit_range_conditional_variance(innovation, distance, state)
