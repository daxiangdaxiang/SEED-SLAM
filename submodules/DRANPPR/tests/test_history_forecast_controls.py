import numpy as np

from experiments.audit_history_forecast import forecast_controls


def test_noiseless_acceleration_and_unobserved_stop_controls():
    rows, _ = forecast_controls(trials=3, noises=(0.,), horizons=(4.,))
    selected = {(r['curve'],r['prediction']):r for r in rows}
    assert selected['quadratic','risk3']['mean_squared_error'] < 1e-20
    assert selected['quadratic','poly2']['mean_squared_error'] < 1e-20
    assert selected['stop','hold']['mean_squared_error'] == 0.
    assert selected['stop','risk3']['mean_squared_error'] > 0.
    assert np.sum(selected['quadratic','risk3']['degree_counts'][2:]) == 3
