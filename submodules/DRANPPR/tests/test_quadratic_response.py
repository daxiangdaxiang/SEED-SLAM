import importlib.util

import numpy as np
import pytest

from experiments.async_interface import Actor, Network, Packet, Scenario, simulate
from experiments.async_interface_model import normalize


def model_class():
    assert importlib.util.find_spec('experiments.quadratic_response'), 'model response not implemented'
    from experiments.quadratic_response import QuadraticResponse
    return QuadraticResponse


@pytest.mark.parametrize('dimension', [1, 2, 6])
@pytest.mark.parametrize('gain', [0., .3, 1.])
def test_exact_frozen_boundary_recurrence(dimension, gain):
    rng = np.random.default_rng(dimension)
    a = rng.normal(size=(dimension, dimension))
    a = a.T@a+np.eye(dimension)
    p = np.linalg.inv(a+2.*np.eye(dimension))
    response = model_class()(a, p*.4, p, period=3)
    x, b = rng.normal(size=dimension), rng.normal(size=dimension)
    response.reset(x)
    response.receive(6, x, a@x+b, gain=gain, fresh=True)
    expected = x.copy()
    for n in range(21):
        np.testing.assert_allclose(response.predict(6+3*n), expected, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(response.predict(8+3*n), expected, atol=3e-13, rtol=3e-13)
        expected -= gain*p@(a@expected+b)


def test_response_saturates_and_rejects_unstable_descriptor():
    response = model_class()(np.eye(2), np.eye(2)*.5, np.eye(2))
    response.reset(np.array([2., 3.]))
    np.testing.assert_array_equal(response.predict(100), [2., 3.])
    response.receive(0, np.array([2., 3.]), np.array([1., -1.]), gain=1., fresh=True)
    np.testing.assert_allclose(response.predict(1000000), [1., 4.], atol=1e-14)
    with pytest.raises(ValueError, match='contractive'):
        model_class()(np.eye(2), np.eye(2)*2., np.eye(2))


def test_signed_forcing_innovation_and_frozen_ablation():
    cls = model_class()
    a = np.array([[2., .5], [.5, 1.]])
    p = np.linalg.inv(a+np.eye(2))
    active, frozen = [cls(a, p, p, correct=c) for c in (True, False)]
    x0, x1 = np.array([1., 2.]), np.array([-2., 4.])
    b0, b1 = np.array([.3, -.4]), np.array([.1, .5])
    for model in (active, frozen):
        model.reset(x0)
        model.receive(0, x0, a@x0+b0)
        model.receive(3, x1, a@x1+b1)
        np.testing.assert_allclose(model.last_innovation, b1-b0, atol=1e-14)
        np.testing.assert_array_equal(model.predict(3), x1)
    np.testing.assert_allclose(active.predict(4), x1-p@(a@x1+b1), atol=1e-14)
    np.testing.assert_allclose(frozen.predict(4), x1-p@(a@x1+b0), atol=1e-14)
    active.reset(x0)
    assert active.forcing is None
    np.testing.assert_array_equal(active.predict(100), x0)


def test_packet_owns_model_payload_and_charges_every_hop():
    assert 'model_gradient' in Packet.__dataclass_fields__, 'model packet not implemented'
    received = []
    network = Network([(0, 1), (1, 2)], Scenario('ideal'), received.append)
    value, gradient = np.array([1., 2.]), np.array([3., 4.])
    packet = Packet(0, 2, 1, 0, value, model_gradient=gradient, model_gain=.7, model_fresh=True)
    network.send(packet, 0)
    value[:] = gradient[:] = 100.
    network.advance(0)
    np.testing.assert_array_equal(received[0].value, [1., 2.])
    np.testing.assert_array_equal(received[0].model_gradient, [3., 4.])
    assert network.application_bytes == 2*(40+16+16+16)


def test_stale_packet_does_not_rewrite_model():
    cls = model_class()
    actor = Actor(0, {0: np.eye(1), 1: np.eye(1)*.2}, np.zeros(1), np.eye(1)*.1)
    actor.models = {1: cls(np.eye(1), np.eye(1)*.5, np.eye(1))}
    actor.models[1].reset(np.zeros(1))
    assert actor.receive(Packet(1, 0, 2, 2, np.ones(1), model_gradient=np.ones(1)))
    prediction = actor.models[1].predict(5)
    assert not actor.receive(Packet(1, 0, 1, 1, np.ones(1)*8., model_gradient=np.ones(1)*9.))
    np.testing.assert_array_equal(actor.models[1].predict(5), prediction)


@pytest.mark.parametrize('prediction', ['model_response', 'model_frozen'])
def test_ideal_parity_causal_prefix_and_extra_setup_bytes(prediction):
    cls = model_class()
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    for name in ['ideal', 'delay', 'heterogeneous']:
        scenario = Scenario(name)
        baseline = simulate(model, [(0, 1)], scenario, 'directional', 150, initial)
        run = simulate(model, [(0, 1)], scenario, 'error_budget', 150, initial, prediction=prediction)
        longer = simulate(model, [(0, 1)], scenario, 'error_budget', 170, initial, prediction=prediction)
        np.testing.assert_array_equal(run['states'], longer['states'][:150])
        np.testing.assert_array_equal(run['counters'][:, 1], baseline['counters'][:, 1])
        assert run['summary']['model_setup_bytes'] == 2*cls.descriptor_bytes(1)
        assert run['summary']['application_bytes'] > baseline['summary']['application_bytes']
        if name == 'ideal':
            np.testing.assert_array_equal(run['states'], baseline['states'])


def test_equal_byte_alignment_does_not_interpolate_states():
    assert importlib.util.find_spec('experiments.audit_model_response'), 'model audit not implemented'
    from experiments.audit_model_response import at_common_bytes
    left = dict(counters=np.array([[100, 1], [210, 2], [320, 3]]), relative_gaps=np.array([.8, .6, .4]))
    right = dict(counters=np.array([[60, 1], [120, 2], [180, 3], [240, 4]]),
                 relative_gaps=np.array([.9, .7, .5, .3]))
    result = at_common_bytes(left, right)
    assert result['budget'] == 240
    assert result['candidate_tick'] == 1 and result['reference_tick'] == 3
    assert result['candidate_gap'] == .6 and result['reference_gap'] == .3


def test_sender_model_gradient_uses_only_received_states(monkeypatch):
    sent = []
    original_send = Network.send

    def record(self, packet, time):
        sent.append(Packet(packet.source, packet.destination, packet.revision, packet.stamp,
                           packet.value.copy(), packet.model_gradient.copy(),
                           packet.model_gain, packet.model_fresh))
        original_send(self, packet, time)

    monkeypatch.setattr(Network, 'send', record)
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    initial = np.array([1., -1.])
    scenario = Scenario('delay')
    run = simulate(model, [(0, 1)], scenario, 'error_budget', 160, initial, prediction='model_response')
    monkeypatch.setattr(Network, 'send', original_send)
    caches = {0: initial[1:].copy(), 1: initial[:1].copy()}
    versions = {0: (0, 0), 1: (0, 0)}

    def deliver(packet):
        version = (packet.revision, packet.stamp)
        if version > versions[packet.destination]:
            caches[packet.destination] = packet.value.copy()
            versions[packet.destination] = version

    network = Network([(0, 1)], scenario, deliver)
    by_tick = {t: [p for p in sent if p.stamp == t] for t in range(160)}
    for tick, state in enumerate(run['states']):
        network.advance(tick)
        for p in by_tick[tick]:
            i, j = p.source, p.destination
            expected = model.b[i, i]*state[i]+model.c[i]+model.b[i, j]*caches[i][0]
            np.testing.assert_allclose(p.model_gradient, [expected], atol=1e-14)
            np.testing.assert_array_equal(p.value, state[model.indices[i]])
            network.send(p, tick)
        network.advance(tick)


def test_independent_byte_replay_detects_tampering():
    from experiments.audit_model_response import audit_bytes
    model = normalize(np.array([[2., -.7], [-.7, 1.]]), np.array([1., -.3]), [0, 1])
    scenario = Scenario('heterogeneous')
    run = simulate(model, [(0, 1)], scenario, 'error_budget', 150, prediction='model_response')
    assert audit_bytes(model, [(0, 1)], scenario, run)['prefix_bytes_exact']
    run['counters'][80, 0] += 8
    with pytest.raises(AssertionError):
        audit_bytes(model, [(0, 1)], scenario, run)


def test_malformed_model_packet_does_not_mutate_state():
    cls = model_class()
    response = cls(np.eye(2), np.eye(2)*.5, np.eye(2))
    response.reset(np.ones(2))
    for gradient, gain in ((None, 1.), (np.ones(3), 1.), (np.ones(2), 1.1)):
        with pytest.raises(ValueError):
            response.receive(1, np.ones(2)*2., gradient, gain=gain)
        np.testing.assert_array_equal(response.value, np.ones(2))
        assert response.forcing is None
