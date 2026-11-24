from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import qr, solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@pytest.mark.parametrize('weight_spread', [0., 12.])
def test_sparse_selected_covariance_matches_dense_qr_with_cross_terms(weight_spread):
    from dranppr_sparse_prediction import sparse_selected_covariance
    rng = np.random.default_rng(219)
    a = rng.normal(size=(90, 30))
    a *= np.exp(np.linspace(-weight_spread/2, weight_spread/2, len(a)))[:, None]
    rhs = np.eye(30)[:, [0, 4, 9, 21, 29]]
    r = qr(a, mode='economic')[1]
    inverse = solve_triangular(r.T, rhs, lower=True)
    expected = inverse.T @ inverse
    actual = sparse_selected_covariance(a, rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11)
    assert abs(actual[0, 1]) > 1e-8


def test_sparse_selected_covariance_keeps_exactly_fixed_query_dimensions_zero():
    from dranppr_sparse_prediction import sparse_selected_covariance
    rng = np.random.default_rng(220)
    a = rng.normal(size=(20, 5))
    rhs = np.column_stack([np.zeros((5, 3)), np.eye(5)[:, :2]])
    covariance = sparse_selected_covariance(a, rhs)
    np.testing.assert_array_equal(covariance[:3], np.zeros((3, 5)))
    np.testing.assert_array_equal(covariance[:, :3], np.zeros((5, 3)))


def test_sparse_selected_covariance_rejects_singular_state_model():
    from dranppr_sparse_prediction import sparse_selected_covariance
    a = np.column_stack([np.ones(10), np.zeros(10)])
    with pytest.raises(ValueError, match='singular'):
        sparse_selected_covariance(a, np.eye(2))
