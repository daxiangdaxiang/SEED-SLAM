import numpy as np
import pytest

from experiments.review_async_interface import coupling_radius, gap_comparison, lyapunov_audit
from experiments.async_interface_model import normalize
from experiments.async_interface import Scenario, simulate


def test_comparison_does_not_rank_machine_precision_as_an_accuracy_gain():
    assert gap_comparison(1e-32, 5e-32) == 'tie'
    assert gap_comparison(.01, .02) == 'win'
    assert gap_comparison(.02, .01) == 'loss'


def test_block_coupling_certificate_is_only_sufficient():
    assert coupling_radius(np.array([[1., -.5], [-.5, 1.]]), [1,1]) == pytest.approx(.5)
    assert coupling_radius(.8*np.ones((5,5))+.2*np.eye(5), [1]*5) > 1.


def test_directional_delayed_lyapunov_bound_with_mixed_block_dimensions():
    rng = np.random.default_rng(53)
    a = rng.normal(size=(16, 8))
    model = normalize(a.T@a, rng.normal(size=8), [0,0,1,1,1,2,2,2])
    initial = np.sin(np.arange(8)+1.)
    run = simulate(model, [(0,1),(1,2)], Scenario('heterogeneous'), 'directional', 120, initial=initial)
    audit = lyapunov_audit(model, run['states'], initial, run['summary']['delay_bounds'])
    assert audit['max_normalized_violation'] < 1e-12
    assert audit['final_energy'] < audit['initial_energy']


def test_energy_audit_rejects_unsafe_delayed_updates():
    model = normalize(.8*np.ones((5,5))+.2*np.eye(5),np.ones(5),list(range(5)))
    initial = np.sin(np.arange(5)+1.)
    links = [(i,j) for i in range(5) for j in range(i+1,5)]
    run = simulate(model,links,Scenario('delay',4,4),'stale',100,initial=initial)
    audit = lyapunov_audit(model,run['states'],initial,run['summary']['delay_bounds'])
    assert audit['max_normalized_violation'] > 1.
