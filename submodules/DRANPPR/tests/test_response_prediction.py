import numpy as np
import pytest

from experiments import neighbor_prediction
from experiments.async_interface import Actor, Packet, Scenario, simulate
from experiments.async_interface_model import normalize


def predictor(mode='response_calibrated', size=8, metric=None):
    assert hasattr(neighbor_prediction, 'ResponsePrediction'), 'response predictor not implemented'
    return neighbor_prediction.ResponsePrediction(mode, size, metric)


@pytest.mark.parametrize('mode', ['response_linear', 'response_calibrated'])
def test_irregular_linear_response_and_exact_cache_anchor(mode):
    p = predictor(mode)
    for t in [0., 1., 2.5, 4., 7., 9.]:
        p.add(t, np.array([3.+2.*t, -4.-t]))
    np.testing.assert_allclose(p.predict(13.), [29., -17.], atol=1e-12)
    np.testing.assert_array_equal(p.predict(9.), [21., -13.])


def test_calibration_scores_old_forecast_before_ingesting_new_value():
    p = predictor()
    p.add(0., np.array([0.]))
    p.add(1., np.array([2.]))
    assert p.gain == 0.
    p.add(3., np.array([4.]))
    # Old velocity=2, observed velocity=1. This must not use the refitted slope.
    assert p.gain == pytest.approx(.5)
    np.testing.assert_allclose(p.last_raw_innovation, [-2.])
    np.testing.assert_allclose(p.last_innovation, [2.])
    assert p.validations == 1


def test_gain_uses_receiver_metric_and_is_clipped():
    p = predictor(metric=np.diag([1., 4.]))
    p.add(0., np.zeros(2))
    p.add(1., np.ones(2))
    p.add(2., np.array([2., 1.]))
    assert p.gain == pytest.approx(.2)
    q = predictor()
    for t, x in enumerate([0., 1., 0.]):
        q.add(t, np.array([x]))
    assert q.gain == 0.


def test_prediction_is_read_only_without_refitting_or_history_feedback():
    p = predictor()
    for t in range(8):
        p.add(float(t), np.array([.3*t]))
    before = (p.fits, p.validations, p.gain, p.sum_pp, p.sum_pd)
    history = np.array(p.values).copy()
    for t in range(8, 30):
        p.predict(t)
    assert before == (p.fits, p.validations, p.gain, p.sum_pp, p.sum_pd)
    np.testing.assert_array_equal(history, p.values)


def test_translation_scale_and_time_offset_invariance():
    a, b = predictor(), predictor()
    for t, x in zip([0., 1., 3., 4., 7., 9.], [1., 2., 3., 5., 6., 6.5]):
        a.add(t, np.array([x]))
        b.add(t+1000., np.array([7.*x+90.]))
    np.testing.assert_allclose(b.predict(1012.), 7.*a.predict(12.)+90., atol=1e-11)
    assert b.gain == pytest.approx(a.gain)


def test_packet_fit_and_gain_match_independent_batch_regressions():
    metric = np.array([[2., .3], [.3, 1.]])
    p = predictor(metric=metric)
    rng = np.random.default_rng(52)
    times = np.cumsum(rng.uniform(.2, 2., 24))
    values = rng.normal(size=(24, 2))
    samples = []
    for index, (time, value) in enumerate(zip(times, values)):
        if p.velocity is not None:
            observed = (value-p.values[-1])/(time-p.times[-1])
            samples.append((p.velocity.copy(), observed))
        p.add(time, value)
        if index:
            t = np.asarray(p.times)-time
            design = np.column_stack([np.ones(len(t)), t])
            slope = np.linalg.lstsq(design, np.asarray(p.values), rcond=None)[0][1]
            np.testing.assert_allclose(p.velocity, slope, atol=1e-12)
        if samples:
            weights = p.discount**np.arange(len(samples)-1, -1, -1)
            pp = sum(w*(v@metric@v) for w, (v, d) in zip(weights, samples))
            pd = sum(w*(v@metric@d) for w, (v, d) in zip(weights, samples))
            assert p.sum_pp == pytest.approx(pp)
            assert p.sum_pd == pytest.approx(pd)
            assert p.gain == pytest.approx(np.clip(pd/pp, 0., 1.))


def test_response_history_validation_and_reset():
    p = predictor()
    for t in range(20):
        p.add(t, np.array([float(t)]))
    assert len(p.times) == 8
    with pytest.raises(ValueError):
        p.add(19., np.ones(1))
    with pytest.raises(ValueError):
        p.add(21., np.array([np.nan]))
    p.reset(30., np.array([-3.]))
    assert p.gain == 0. and p.validations == 0
    np.testing.assert_array_equal(p.predict(40.), [-3.])


@pytest.mark.parametrize('mode', ['response_linear', 'response_calibrated'])
def test_actor_response_keeps_actual_cache_rejects_old_and_uses_only_coupled_space(mode):
    actor = Actor(0, {0: np.eye(1), 1: np.array([[.5, 0.]])}, np.zeros(1), .5, prediction=mode)
    for t in range(1, 5):
        actor.receive(Packet(1, 0, t, t, np.array([float(t), 1000.*(-1)**t])))
    before = actor.cache[1].copy()
    np.testing.assert_allclose(actor.gradient(tick=7), [3.], atol=1e-12)
    np.testing.assert_array_equal(actor.cache[1], before)
    assert not actor.receive(Packet(1, 0, 2, 2, np.zeros(2)))


@pytest.mark.parametrize('mode', ['response_linear', 'response_calibrated'])
def test_response_ideal_parity_delayed_prefix_and_budget(mode):
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    for net in ['ideal', 'delay', 'outage']:
        scenario = Scenario(net, outage=(30, 60))
        reference = simulate(model, [(0, 1)], scenario, 'directional', 100, initial)
        candidate = simulate(model, [(0, 1)], scenario, 'directional', 100, initial, prediction=mode)
        np.testing.assert_array_equal(reference['counters'], candidate['counters'])
        if net == 'ideal':
            np.testing.assert_array_equal(reference['states'], candidate['states'])
        longer = simulate(model, [(0, 1)], scenario, 'directional', 110, initial, prediction=mode)
        np.testing.assert_array_equal(candidate['states'], longer['states'][:100])


def test_offline_exact_neighbor_diagnostic_retains_metric_and_ideal_parity():
    from experiments import audit_response_controls as controls
    assert hasattr(controls, 'exact_neighbor_diagnostic')
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.sin(np.arange(2)+1.)
    reference = simulate(model, [(0, 1)], Scenario('ideal'), 'directional', 100, initial)
    diagnostic = controls.exact_neighbor_diagnostic(model, [(0, 1)], Scenario('ideal'), 100)
    np.testing.assert_allclose(diagnostic['states'], reference['states'], atol=1e-14)
    assert diagnostic['deployable'] is False


def test_controls_retain_unobservable_change_and_explicit_post_message_recovery():
    from experiments.audit_response_controls import forecasts
    result = forecasts(2)
    assert len(result['cells']) == 270
    rows = result['stop_recovery']
    for mode in ['response_linear', 'response_calibrated']:
        selected = [x for x in rows if x['prediction'] == mode]
        assert selected[0]['squared_error'] == pytest.approx(.008)
        assert selected[-1]['squared_error'] == 0.
    a = next(x for x in rows if x['prediction'] == 'response_linear' and x['received_post_change'] == 4)
    b = next(x for x in rows if x['prediction'] == 'response_calibrated' and x['received_post_change'] == 4)
    assert b['squared_error'] < a['squared_error']


def test_review_does_not_count_ideal_ties_as_nonideal_improvements():
    from experiments.review_response_prediction import aggregate
    items = [dict(prediction='response_calibrated', reference='hold', scenario=net,
                  case='d1', gap_result=outcome, gap_reduction=reduction)
             for net, outcome, reduction in [('delay', 'loss', -.2), ('ideal', 'tie', 0.)]]
    result = aggregate(items)
    assert result['response_calibrated_vs_hold:all']['gap'] == dict(win=0, tie=1, loss=1)
    assert result['response_calibrated_vs_hold:nonideal']['median_gap_reduction'] == -.2
