import numpy as np
import pytest

from experiments.async_interface_model import eliminate, normalize
from experiments.async_interface import Actor, Packet, Network, Scenario, simulate, delay_bounds
from experiments.async_interface import directional_majorizer


def example():
    h = np.array([[3., -.8, .2], [-.8, 2., -.7], [.2, -.7, 1.]])
    return normalize(h, np.array([1., -2., .3]), [0, 1, 2])


def test_elimination_matches_full_system_and_recovery():
    rng = np.random.default_rng(31)
    a = rng.normal(size=(19, 8))
    b = rng.normal(size=19)
    reduced = eliminate(a[:, :5], a[:, 5:], b)
    x = np.linalg.solve(reduced.h, -reduced.g)
    p = reduced.recover(x)
    np.testing.assert_allclose(np.r_[p, x], np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-12)
    assert reduced.private_residual(x) < 1e-12


def test_singular_private_or_global_models_are_not_regularized():
    with pytest.raises(ValueError, match='private'):
        eliminate(np.zeros((4, 2)), np.eye(4)[:, :2], np.ones(4))
    with pytest.raises(ValueError, match='positive definite'):
        normalize(np.ones((2, 2)), np.ones(2), [0, 1])


def test_curvature_coordinates_preserve_objective():
    model = example()
    y = np.array([.2, -.4, .5])
    x = model.to_physical(y)
    assert .5*x@model.h@x + model.g@x == pytest.approx(.5*y@model.b@y+model.c@y)
    for indices in model.indices.values():
        np.testing.assert_allclose(model.b[np.ix_(indices, indices)], np.eye(len(indices)), atol=1e-12)
    assert model.condition > 1


def test_actor_uses_only_received_state_and_rejects_stale_or_duplicate():
    actor = Actor(0, {0: np.eye(1), 1: np.array([[.5]])}, np.array([1.]), .5)
    first = Packet(1, 0, 3, 10, np.array([2.]))
    assert actor.receive(first)
    assert not actor.receive(Packet(1, 0, 2, 11, np.array([99.])))
    assert not actor.receive(first)
    first.value[:] = 20.
    np.testing.assert_allclose(actor.gradient(), [2.])
    actor.update()
    np.testing.assert_allclose(actor.value, [-1.])
    assert actor.revision == 1


def test_route_counts_each_hop_and_delivers_no_earlier_than_due():
    delivered = []
    network = Network([(0, 1), (1, 2)], Scenario('delay', 1, 1), delivered.append)
    packet = Packet(0, 2, 1, 0, np.ones(2))
    network.send(packet, 0)
    network.advance(1)
    assert not delivered
    network.advance(2)
    assert len(delivered) == 1
    assert network.application_bytes == 2*(40+8*2)
    assert network.hop_messages == 2
    with pytest.raises(ValueError, match='connected'):
        Network([(0, 1), (2, 3)], Scenario('ideal'), delivered.append)


def test_outage_contract_only_penalizes_routes_crossing_failed_link():
    links = [(0, 1), (1, 2), (2, 3), (3, 4)]
    periods = {i: 1 for i in range(5)}
    delayed = Network(links, Scenario('delay'), lambda packet: None)
    outage = Network(links, Scenario('outage'), lambda packet: None)
    a, b = [delay_bounds(n, list(periods), periods) for n in (delayed, outage)]
    assert a[0,1] == b[0,1]
    assert b[0,4]-a[0,4] == 100


def test_zero_delay_synchronous_asynchronous_parity():
    model = example()
    runs = [simulate(model, [(0, 1), (1, 2)], Scenario('ideal'), mode, 80)
            for mode in ('synchronous', 'stale', 'bounded')]
    for run in runs[1:]:
        np.testing.assert_array_equal(run['states'], runs[0]['states'])
        assert run['summary']['application_bytes'] == runs[0]['summary']['application_bytes']
    assert runs[0]['summary']['relative_residual'] < 1e-6


def test_nonzero_common_start_changes_initial_state_not_objective():
    model = example()
    initial = np.array([1., -.5, .25])
    saved = initial.copy()
    run = simulate(model, [(0, 1), (1, 2)], Scenario('ideal'), 'stale', 1, initial=initial)
    steps = np.array([run['summary']['steps'][i] for i in range(3)])
    np.testing.assert_allclose(run['states'][0], initial-steps*(model.b@initial+model.c))
    np.testing.assert_array_equal(initial, saved)
    assert run['summary']['initial_state_norm'] == pytest.approx(np.linalg.norm(initial))


@pytest.mark.parametrize('name', ['delay', 'heterogeneous', 'loss', 'outage'])
def test_declared_bounds_and_recovery(name):
    model = example()
    scenario = Scenario(name, delay_min=1, delay_max=3, outage=(30, 60))
    run = simulate(model, [(0, 1), (1, 2)], scenario, 'bounded', 2500)
    assert run['summary']['bound_violations'] == 0
    assert run['summary']['relative_gap'] < 1e-3
    assert run['summary']['global_reductions'] == 0
    assert run['summary']['updates'] > 0
    assert np.isfinite(run['states']).all()


def test_delivery_is_causal_and_continuation_preserves_prefix():
    model = example()
    short = simulate(model, [(0, 1), (1, 2)], Scenario('loss', seed=3), 'stale', 50)
    long = simulate(model, [(0, 1), (1, 2)], Scenario('loss', seed=3), 'stale', 90)
    np.testing.assert_array_equal(short['states'], long['states'][:50])
    assert short['summary']['dropped_packets'] > 0


def test_strong_spd_coupling_is_not_a_blanket_asynchrony_guarantee():
    h = .8*np.ones((5, 5))+.2*np.eye(5)
    model = normalize(h, np.ones(5), list(range(5)))
    links = [(i, i+1) for i in range(4)]
    scenario = Scenario('delay', delay_min=4, delay_max=4)
    naive = simulate(model, links, scenario, 'stale', 3000)
    safe = simulate(model, links, scenario, 'bounded', 3000)
    assert safe['summary']['relative_gap'] < 1e-3
    assert safe['summary']['bound_violations'] == 0
    # The naive result is a measured diagnostic, not assumed to diverge for
    # every topology or delay pattern.
    assert naive['summary']['updates'] == safe['summary']['updates']


def test_uniform_delayed_spd_counterexample_requires_algorithmic_control():
    model = normalize(.8*np.ones((5, 5))+.2*np.eye(5), np.ones(5), list(range(5)))
    links = [(i, j) for i in range(5) for j in range(i+1, 5)]
    scenario = Scenario('delay', delay_min=4, delay_max=4)
    initial = np.sin(np.arange(5)+1.)
    naive = simulate(model, links, scenario, 'stale', 1000, initial=initial)
    bounded = simulate(model, links, scenario, 'bounded', 1000, initial=initial)
    assert naive['summary']['relative_gap'] > 1e20
    assert bounded['summary']['relative_gap'] < 1e-3
    np.testing.assert_array_equal(naive['counters'][:, 0], bounded['counters'][:, 0])


def test_directional_delay_control_preserves_uncoupled_direction():
    rows = {0: np.eye(2), 1: np.diag([.4, 0.])}
    bounds = {(0,1): 6, (1,0): 6}
    metric = directional_majorizer(rows, 0, bounds)
    np.testing.assert_allclose(metric, np.diag([6.2, 1.]))
    actor = Actor(0, rows, np.ones(2), np.linalg.inv(metric))
    actor.update()
    np.testing.assert_allclose(actor.value, [-1./6.2, -1.])


def test_directional_cross_bound_and_scalar_domination():
    rng = np.random.default_rng(42)
    cross = rng.normal(size=(3, 5))
    u, s, vt = np.linalg.svd(cross, full_matrices=False)
    left, right = (u*s)@u.T, (vt.T*s)@vt
    bound = np.block([[left, -cross], [-cross.T, right]])
    assert np.linalg.eigvalsh(bound).min() > -1e-12
    metric = directional_majorizer({0:np.eye(3),1:cross}, 0, {(0,1):3,(1,0):5})
    scalar = (1.+9.*np.linalg.norm(cross,2))*np.eye(3)
    assert np.linalg.eigvalsh(scalar-metric).min() > -1e-12


def test_directional_mode_matches_scalar_for_scalar_blocks():
    model = example()
    scenario = Scenario('delay')
    reference = simulate(model, [(0,1),(1,2)], scenario, 'bounded', 100)
    candidate = simulate(model, [(0,1),(1,2)], scenario, 'directional', 100)
    np.testing.assert_allclose(candidate['states'], reference['states'], atol=1e-14)
    np.testing.assert_array_equal(candidate['counters'], reference['counters'])
