import numpy as np
import pytest

from experiments.neighbor_prediction import HistoryPrediction
from experiments.async_interface import Actor, Packet, Scenario, simulate
from experiments.async_interface_model import normalize


@pytest.mark.parametrize('degree', [1, 2, 3])
def test_irregular_polynomial_is_reproduced(degree):
    times = np.array([0., 1., 2.5, 4., 5., 7., 8., 11.])
    def curve(t):
        return np.array([sum(.01**k*t**k for k in range(degree+1)), -2.*t])
    predictor = HistoryPrediction('poly'+str(degree), 8)
    for t in times:
        predictor.add(t, curve(t))
    np.testing.assert_allclose(predictor.predict(13.), curve(13.), atol=1e-11)
    np.testing.assert_array_equal(predictor.predict(11.), curve(11.))


def test_history_is_copied_ordered_bounded_and_resettable():
    predictor = HistoryPrediction('poly2', 8)
    value = np.array([1.])
    predictor.add(0., value)
    value[:] = 99.
    np.testing.assert_array_equal(predictor.predict(1.), [1.])
    with pytest.raises(ValueError, match='time'):
        predictor.add(0., value)
    for i in range(1, 20):
        predictor.add(float(i), np.array([float(i)]))
    assert len(predictor.times) == 8
    predictor.reset(30., np.array([-2.]))
    np.testing.assert_array_equal(predictor.predict(40.), [-2.])


def test_history_linear_fit_reduces_noise_vs_two_point_velocity():
    rng = np.random.default_rng(42)
    errors = {mode: [] for mode in ('secant', 'poly1')}
    for _ in range(200):
        values = np.arange(8.)[:, None]+rng.normal(0., .2, (8, 1))
        for mode in errors:
            predictor = HistoryPrediction(mode, 8)
            for t, value in enumerate(values):
                predictor.add(float(t), value)
            errors[mode].append(float(predictor.predict(12.)[0]-12.))
    assert np.mean(np.square(errors['poly1'])) < .3*np.mean(np.square(errors['secant']))


def test_predictor_actor_never_changes_received_cache():
    actor = Actor(0, {0: np.eye(1), 1: np.array([[.5]])}, np.array([1.]), .5,
                  prediction='poly2')
    actor.receive(Packet(1, 0, 1, 1, np.array([2.])))
    actor.receive(Packet(1, 0, 2, 2, np.array([3.])))
    saved = actor.cache[1].copy()
    actor.gradient(tick=7)
    np.testing.assert_array_equal(actor.cache[1], saved)
    assert not actor.receive(Packet(1, 0, 1, 1, np.array([99.])))
    assert actor.histories[1].times[-1] == 2


@pytest.mark.parametrize('mode', ['secant', 'poly1', 'poly2', 'poly3'])
def test_prediction_zero_delay_parity_and_same_prefix_budget(mode):
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    reference = simulate(model, [(0,1)], Scenario('ideal'), 'directional', 80, initial)
    candidate = simulate(model, [(0,1)], Scenario('ideal'), 'directional', 80, initial,
                         prediction=mode)
    np.testing.assert_array_equal(reference['states'], candidate['states'])
    delayed = simulate(model, [(0,1)], Scenario('delay'), 'directional', 80, initial)
    predicted = simulate(model, [(0,1)], Scenario('delay'), 'directional', 80, initial,
                         prediction=mode)
    np.testing.assert_array_equal(delayed['counters'], predicted['counters'])
    assert predicted['gradient_error_sq'].shape == (80, 2)
    longer = simulate(model, [(0,1)], Scenario('delay'), 'directional', 90, initial,
                      prediction=mode)
    np.testing.assert_array_equal(predicted['states'], longer['states'][:80])


def test_prediction_review_recomputes_metrics_and_detects_budget_change():
    from experiments.audit_neighbor_prediction import review_pair
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    baseline = simulate(model, [(0,1)], Scenario('delay'), 'directional', 80, initial)
    candidate = simulate(model, [(0,1)], Scenario('delay'), 'directional', 80, initial, prediction='poly2')
    result = review_pair(model, initial, candidate, baseline)
    assert result['prefix_budget_equal']
    assert result['candidate_gap'] == pytest.approx(candidate['relative_gaps'][-1])
    candidate['counters'][20,0] += 1
    with pytest.raises(ValueError, match='budget'):
        review_pair(model, initial, candidate, baseline)


def test_risk_history_reproduces_supported_curvature_and_keeps_cache_anchor():
    predictor = HistoryPrediction('risk3', 8)
    for t in range(8):
        predictor.add(t, np.array([.01*t**3+.2*t, -.03*t*t]))
    np.testing.assert_allclose(predictor.predict(10.), [12., -3.], atol=1e-11)
    np.testing.assert_array_equal(predictor.predict(7.), [4.83, -1.47])


def test_risk_history_does_not_promote_unsupported_noise_to_cubic_trend():
    rng = np.random.default_rng(31)
    errors = {mode: [] for mode in ('poly3','risk3')}
    for _ in range(200):
        values = np.arange(8.)[:,None]*np.array([[.02,.01]])+rng.normal(0.,.03,(8,2))
        for mode in errors:
            predictor = HistoryPrediction(mode,8)
            for t,value in enumerate(values):
                predictor.add(float(t),value)
            errors[mode].append(np.sum((predictor.predict(12.)-np.array([.24,.12]))**2))
    assert np.mean(errors['risk3']) < .2*np.mean(errors['poly3'])


def test_coupling_weighted_risk_ignores_irrelevant_direction_noise():
    predictor = HistoryPrediction('risk3',8, influence=np.array([[1.,0.]]))
    rng = np.random.default_rng(4)
    for t in range(8):
        predictor.add(t, np.array([float(t), rng.normal(0.,100.)]))
    assert predictor.predict(10.)[0] == pytest.approx(10.)


def test_risk_simulation_is_causal_and_preserves_zero_delay():
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0,1])
    initial = np.array([1.,-1.])
    reference = simulate(model, [(0,1)], Scenario('ideal'), 'directional',80,initial)
    candidate = simulate(model, [(0,1)], Scenario('ideal'), 'directional',80,initial,prediction='risk3')
    np.testing.assert_array_equal(reference['states'],candidate['states'])


def test_metric_review_handles_matrix_product_roundoff_near_stationarity():
    from experiments.audit_neighbor_prediction import review_pair
    model = normalize(.8*np.ones((5,5))+.2*np.eye(5),np.ones(5),list(range(5)))
    initial = np.sin(np.arange(5)+1.)
    run = simulate(model,[(0,1),(1,2),(2,3),(3,4)],Scenario('ideal'),'directional',2000,initial)
    assert run['relative_residuals'][-1] < 1e-14
    review_pair(model, initial, run, run)


def test_posthoc_gradient_error_uses_preupdate_state():
    model = normalize(np.array([[2.,-.7],[-.7,1.]]),np.array([1.,-.3]),[0,1])
    initial = np.array([1.,-1.])
    run = simulate(model,[(0,1)],Scenario('delay',3,3),'directional',2,initial)
    errors = []
    for i in (0,1):
        j=1-i
        errors.append(model.b[i,j]*(initial[j]-run['states'][0,j]))
    np.testing.assert_allclose(run['gradient_error_sq'][1],np.sum(np.square(errors)))


def test_step_audit_reconstructs_prediction_error_and_conditional_descent_bound():
    from experiments.review_history_prediction import inexact_gradient_audit
    model = normalize(np.array([[2.,-.7],[-.7,1.]]),np.array([1.,-.3]),[0,1])
    initial = np.array([1.,-1.])
    run = simulate(model,[(0,1)],Scenario('delay'),'directional',100,initial,prediction='risk3')
    result = inexact_gradient_audit(model,initial,run['states'],run['summary'])
    assert result['max_normalized_bound_violation'] < 1e-12
    assert result['reconstructed_error_sq_sum'] == pytest.approx(run['gradient_error_sq'][:,0].sum())
