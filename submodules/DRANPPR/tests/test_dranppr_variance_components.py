from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_contrast_likelihood_is_invariant_to_state_and_matches_direct_gaussian():
    from dranppr_variance_components import ContrastVarianceModel
    rng = np.random.default_rng(31)
    a = rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    d = np.column_stack([np.arange(30) % 2 == 0, np.arange(30) % 2 == 1]).astype(float)
    model = ContrastVarianceModel(a, y, d, np.zeros(30))
    changed = ContrastVarianceModel(a, y + a @ rng.normal(size=5), d, np.zeros(30))
    q = np.array([.4, 2.])
    assert model.nll(np.log(q)) == pytest.approx(changed.nll(np.log(q)), abs=1e-11)
    c = model.contrasts
    covariance = c.T @ np.diag(d @ q) @ c
    z = c.T @ y
    expected = .5 * (z @ np.linalg.solve(covariance, z) +
                     np.linalg.slogdet(covariance)[1] + len(z)*np.log(2*np.pi))
    assert model.nll(np.log(q)) == pytest.approx(expected, abs=1e-11)


def test_contrast_score_gradient_matches_finite_differences():
    from dranppr_variance_components import ContrastVarianceModel
    rng = np.random.default_rng(32)
    a, y = rng.normal(size=(25, 4)), rng.normal(size=25)
    components = rng.uniform(.1, 2., size=(25, 3))
    model = ContrastVarianceModel(a, y, components, np.full(25, .1))
    theta = np.log([.3, .7, 2.])
    eps = 1e-5
    numerical = [(model.nll(theta + eps*e)-model.nll(theta-eps*e))/(2*eps)
                 for e in np.eye(3)]
    np.testing.assert_allclose(model.gradient(theta), numerical, rtol=1e-7, atol=1e-8)
    assert np.linalg.eigvalsh(model.fisher(theta)).min() > 0


def test_unidentifiable_variance_components_are_reported_not_fitted():
    from dranppr_variance_components import ContrastVarianceModel
    model = ContrastVarianceModel(np.ones((10, 1)), np.arange(10.),
                                  np.ones((10, 2)), np.zeros(10))
    assert model.component_rank == 1
    with pytest.raises(ValueError, match='identifiable'):
        model.fit(np.zeros(2))


def test_single_scale_recovers_residual_degrees_of_freedom_estimate():
    from dranppr_variance_components import ContrastVarianceModel
    rng = np.random.default_rng(33)
    a, y = rng.normal(size=(80, 8)), rng.normal(size=80)
    model = ContrastVarianceModel(a, y, np.ones((80, 1)), np.zeros(80))
    result = model.fit(np.zeros(1))
    expected = np.sum((model.contrasts.T @ y)**2) / 72
    assert result['success']
    assert result['components'][0] == pytest.approx(expected, rel=1e-6)
