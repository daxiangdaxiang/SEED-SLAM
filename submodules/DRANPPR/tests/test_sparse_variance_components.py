from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def problem():
    rng = np.random.default_rng(317)
    a = rng.normal(size=(44, 9))
    a[rng.uniform(size=a.shape) < .55] = 0.
    return a, rng.normal(size=44), rng.uniform(.1, 1., size=(44, 3)), np.full(44, .12)


@pytest.mark.parametrize('chunk', [1, 7, 128])
def test_sparse_reml_matches_dense_contrasts_and_chunk_is_only_memory(chunk):
    from dranppr_variance_components import ContrastVarianceModel
    from dranppr_sparse_variance_components import SparseVarianceModel
    a, y, d, base = problem()
    reference = ContrastVarianceModel(a, y, d, base)
    model = SparseVarianceModel(sparse.csr_matrix(a), y, d, base, chunk_size=chunk)
    theta = np.log([.3, 2., .7])
    assert model.nll(theta) == pytest.approx(reference.nll(theta), rel=0., abs=1e-10)
    np.testing.assert_allclose(model.gradient(theta), reference.gradient(theta), rtol=0., atol=1e-10)
    np.testing.assert_allclose(model.fisher(theta), reference.fisher(theta), rtol=0., atol=1e-10)
    assert model.max_solve_residual < 1e-10


def test_sparse_reml_is_invariant_to_state_and_column_units():
    from dranppr_sparse_variance_components import SparseVarianceModel
    a, y, d, base = problem()
    theta = np.log([.3, 2., .7])
    before = SparseVarianceModel(a, y, d, base)
    after = SparseVarianceModel(a * np.logspace(-5, 5, a.shape[1]),
                                y + a @ np.arange(a.shape[1]), d, base)
    assert before.nll(theta) == pytest.approx(after.nll(theta), rel=0., abs=1e-9)
    np.testing.assert_allclose(before.gradient(theta), after.gradient(theta), rtol=0., atol=1e-9)


def test_sparse_reml_gradient_is_a_derivative_not_only_reference_agreement():
    from dranppr_sparse_variance_components import SparseVarianceModel
    model = SparseVarianceModel(*problem())
    theta, eps = np.array([-.5, .4, 1.]), 1e-5
    finite = [(model.nll(theta+eps*e)-model.nll(theta-eps*e))/(2*eps) for e in np.eye(3)]
    np.testing.assert_allclose(model.gradient(theta), finite, rtol=1e-7, atol=1e-8)


def test_sparse_preserves_likelihood_for_weak_but_identifiable_geometry():
    from dranppr_variance_components import ContrastVarianceModel
    from dranppr_sparse_variance_components import SparseVarianceModel
    rng = np.random.default_rng(318)
    u, _ = np.linalg.qr(rng.normal(size=(80, 12)))
    v, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    a = (u*np.logspace(0, -6, 12)) @ v.T
    y = rng.normal(size=80)
    d = np.eye(2)[np.arange(80) % 2]
    dense = ContrastVarianceModel(a, y, d, np.zeros(80))
    model = SparseVarianceModel(a, y, d, np.zeros(80))
    theta = np.log([.3, 2.])
    assert model.nll(theta) == pytest.approx(dense.nll(theta), rel=0., abs=1e-7)
    np.testing.assert_allclose(model.gradient(theta), dense.gradient(theta), rtol=0., atol=1e-6)


def test_sparse_single_scale_fit_keeps_residual_degrees_of_freedom():
    from dranppr_sparse_variance_components import SparseVarianceModel
    a, y, _, _ = problem()
    model = SparseVarianceModel(a, y, np.ones((len(y), 1)), np.zeros(len(y)))
    fit = model.fit(np.zeros(1))
    residual = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    assert fit['success']
    assert fit['components'][0] == pytest.approx(residual @ residual / 35, rel=1e-6)


def test_sparse_rejects_unidentified_state_and_noise_without_regularizing():
    from dranppr_sparse_variance_components import SparseVarianceModel
    with pytest.raises(ValueError, match='rank|singular'):
        SparseVarianceModel(np.ones((10, 2)), np.arange(10.), np.ones((10, 1)), np.zeros(10))
    model = SparseVarianceModel(np.ones((10, 1)), np.arange(10.), np.ones((10, 2)), np.zeros(10))
    with pytest.raises(ValueError, match='identifiable'):
        model.fit(np.zeros(2))


@pytest.mark.parametrize('change', ['zero_column', 'negative_variance', 'nonfinite', 'no_dof'])
def test_sparse_rejects_invalid_models(change):
    from dranppr_sparse_variance_components import SparseVarianceModel
    a, y, d, base = problem()
    if change == 'zero_column':
        a[:, 0] = 0.
    elif change == 'negative_variance':
        base[0] = -1.
    elif change == 'nonfinite':
        y[0] = np.nan
    else:
        a = np.eye(len(y))
    with pytest.raises(ValueError):
        SparseVarianceModel(a, y, d, base)
