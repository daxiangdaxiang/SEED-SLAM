import importlib.util
import json

import numpy as np
import pytest

from experiments.async_interface import Scenario, simulate
from experiments.async_interface_model import normalize


def module():
    assert importlib.util.find_spec('experiments.prediction_error_budget'), 'error budget not implemented'
    from experiments import prediction_error_budget
    return prediction_error_budget


def memory(size=2, max_age=10):
    return module().HorizonErrorMemory(np.eye(1), size, max_age)


def test_only_exact_age_with_enough_prequential_labels_is_supported():
    m = memory()
    m.seed(0, 0, np.array([0.]))
    m.record(2, 0, np.array([1.]))
    m.observe(1, 2, np.array([3.]))
    assert m.bound(2) is None
    m.record(4, 2, np.array([4.]))
    m.observe(2, 4, np.array([5.]))
    assert m.bound(2) == pytest.approx(2.)
    assert m.bound(1) is None
    assert m.bound(3) is None
    assert m.bound(0) == 0.


def test_consecutive_revisions_certify_constant_intervals_not_gaps():
    m = memory(size=1)
    m.seed(0, 0, np.array([2.]))
    m.record(1, 0, np.array([2.5]))
    m.observe(1, 2, np.array([8.]))
    assert m.bound(1) == pytest.approx(.5)
    m.record(3, 2, np.array([20.]))
    m.observe(3, 4, np.array([9.]))
    assert m.unlabelled == 1
    assert m.bound(1) == pytest.approx(.5)


def test_current_label_cannot_validate_its_own_bound():
    m = memory(size=1)
    m.seed(0, 0, np.zeros(1))
    m.record(1, 0, np.zeros(1))
    m.observe(1, 1, np.ones(1))
    assert m.scored_bounds == 0
    m.record(2, 1, np.ones(1))
    m.observe(2, 2, np.array([5.]))
    assert m.scored_bounds == 1 and m.bound_violations == 1
    assert m.bound(1) == 4.


def test_forecast_memory_is_bounded_copied_and_resettable():
    m = memory(size=2, max_age=4)
    m.seed(0, 0, np.zeros(1))
    value = np.ones(1)
    m.record(1, 0, value)
    value[:] = 100.
    m.observe(1, 1, np.ones(1))
    assert m.errors[1][-1] == 0.
    for target in range(2, 20):
        m.record(target, 1, np.ones(1))
    assert len(m.pending) <= 5
    assert m.expired > 0
    m.seed(30, 20, np.array([4.]))
    assert not m.pending and not m.errors


def test_robust_step_has_exact_worst_case_quadratic_bound():
    fn = module().robust_step
    metric = np.array([[2., .4], [.4, 1.]])
    inverse = np.linalg.inv(metric)
    gradient = np.array([1., -2.])
    norm = np.sqrt(gradient@inverse@gradient)
    for fraction in [0., .2, .9, 1., 2.]:
        epsilon = fraction*norm
        step, alpha = fn(gradient, inverse, epsilon)
        assert alpha == pytest.approx(max(0., 1.-fraction))
        true_gradient = gradient-fraction*gradient
        change = true_gradient@step+.5*step@metric@step
        assert change == pytest.approx(-.5*max(0., norm-epsilon)**2, abs=1e-12)


def test_actor_never_reuses_a_planned_update():
    from experiments.async_interface import Actor
    actor = Actor(0, {0: np.eye(1)}, np.ones(1), np.eye(1)*.1,
                  fresh_step=np.eye(1), error_age_limits={})
    gradient = actor.gradient(tick=0)
    actor.update(gradient)
    np.testing.assert_array_equal(actor.value, [-1.])
    actor.update()
    np.testing.assert_array_equal(actor.value, [-1.])


def test_update_without_explicit_gradient_uses_original_cache_policy():
    from experiments.async_interface import Actor
    actor = Actor(0, {0: np.eye(1)}, np.ones(1), np.eye(1)*.1,
                  fresh_step=np.eye(1), error_age_limits={})
    actor.gradient(tick=0)
    actor.update()
    np.testing.assert_array_equal(actor.value, [-.1])


def test_larger_ideal_blocks_preserve_bitwise_legacy_order():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(30, 20))
    model = normalize(a.T@a+np.eye(20), rng.normal(size=20), np.repeat(np.arange(4), 5).tolist())
    initial = np.sin(np.arange(20)+1.)
    links = [(0, 1), (1, 2), (2, 3)]
    baseline = simulate(model, links, Scenario('ideal'), 'directional', 80, initial)
    candidate = simulate(model, links, Scenario('ideal'), 'error_budget', 80, initial)
    np.testing.assert_array_equal(candidate['states'], baseline['states'])


@pytest.mark.parametrize('prediction', ['hold', 'response_linear'])
def test_adaptive_update_is_causal_budget_matched_and_ideal_exact(prediction):
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    for net in ['ideal', 'delay', 'heterogeneous']:
        scenario = Scenario(net)
        baseline = simulate(model, [(0, 1)], scenario, 'directional', 180, initial)
        candidate = simulate(model, [(0, 1)], scenario, 'error_budget', 180, initial, prediction=prediction)
        longer = simulate(model, [(0, 1)], scenario, 'error_budget', 200, initial, prediction=prediction)
        np.testing.assert_array_equal(candidate['counters'], baseline['counters'])
        np.testing.assert_array_equal(candidate['states'], longer['states'][:180])
        if net == 'ideal':
            np.testing.assert_array_equal(candidate['states'], baseline['states'])
        assert candidate['error_budget_trace'].shape == (180, 2, 3)
        assert candidate['summary']['error_budget_statistics']['supported_steps'] > 0
        json.dumps(candidate['summary'], allow_nan=False)


def test_independent_update_audit_detects_modified_dynamic_steps():
    from experiments.audit_error_budget import audit_updates
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    run = simulate(model, [(0, 1)], Scenario('delay'), 'error_budget', 180, initial,
                   prediction='response_linear')
    checked = audit_updates(model, run, initial)
    assert checked['supported_updates'] > 0
    assert checked['conditional_bound_violation'] < 1e-12
    run['states'][100, 0] += .01
    with pytest.raises(ValueError, match='declared local policy'):
        audit_updates(model, run, initial)


def test_audit_summary_supports_selected_ablation_arms():
    from experiments.audit_error_budget import summarize
    rows = [dict(case='small', scenario='delay', seed=1, arm=arm,
                 relative_gap=gap, horizon_complete=True) for arm, gap in
            [('baseline', 1.), ('budget_hold', .5), ('budget_predict', .2)]]
    result = summarize(rows)
    assert result['comparisons']['budget_predict_vs_budget_hold']['win'] == 1
    assert 'budget_predict_vs_prediction_only' not in result['comparisons']
