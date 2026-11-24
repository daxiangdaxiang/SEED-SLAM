import numpy as np

from experiments.async_interface import Scenario, simulate
from experiments.async_interface_model import normalize


def test_coverage_uses_forward_error_scale_near_cancellation():
    from experiments.review_error_budget import error_coverage
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 9))
    model = normalize(a.T@a+np.eye(9), rng.normal(size=9)*1e6,
                      np.repeat(np.arange(3), 3).tolist())
    initial = np.linalg.solve(model.b, -model.c)+.01
    run = simulate(model, [(0, 1), (1, 2)], Scenario('ideal'),
                   'error_budget', 100, initial)
    result = error_coverage(model, initial, run)
    assert result['covered'] == result['supported'] == 300
    assert result['moving_covered'] == result['moving']
    from experiments.audit_error_budget import audit_updates
    assert audit_updates(model, run, initial)['actual_error_covered'] == 300
    run['estimated_gradients'][20, 0] += 1.
    changed = error_coverage(model, initial, run)
    assert changed['covered'] < result['covered']


def test_coverage_does_not_count_zero_steps_as_moving():
    from experiments.review_error_budget import error_coverage
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    run = simulate(model, [(0, 1)], Scenario('delay'), 'error_budget',
                   250, initial, prediction='response_linear')
    result = error_coverage(model, initial, run)
    trace = run['error_budget_trace']
    assert result['moving'] == int(((trace[:, :, 0] > 0) & (trace[:, :, 2] > 0)).sum())
    assert result['moving'] < result['supported']
