import numpy as np
import pytest


@pytest.mark.parametrize("component_count", (1, 2, 7, 19))
def test_factor_matches_dense_solve_and_logdet(component_count):
    from scripts.dba_kronecker_precision import KroneckerPrecisionFactor

    rng = np.random.default_rng(1200 + component_count)
    basis = rng.normal(size=(component_count, component_count))
    component_precision = basis @ basis.T
    prior_precision = np.array([[1.7, 0.35], [0.35, 0.9]])
    information = rng.normal(size=2 * component_count)
    dense = np.kron(component_precision, np.eye(2))
    dense += np.kron(np.eye(component_count), prior_precision)

    factor = KroneckerPrecisionFactor.build(
        component_precision, prior_precision
    )

    assert factor.solve(information) == pytest.approx(
        np.linalg.solve(dense, information), rel=1e-11, abs=1e-11
    )
    sign, logdet = np.linalg.slogdet(dense)
    assert sign == 1.0
    assert factor.half_log_determinant == pytest.approx(
        0.5 * logdet, rel=1e-12, abs=1e-12
    )


def test_factor_represents_the_exact_empty_marginal():
    from scripts.dba_kronecker_precision import KroneckerPrecisionFactor

    factor = KroneckerPrecisionFactor.build(
        np.empty((0, 0)), np.array([[1.7, 0.35], [0.35, 0.9]])
    )

    assert factor.half_log_determinant == 0.0
    assert factor.solve(np.empty(0)).shape == (0,)


def test_factor_removes_only_roundoff_scale_precision_asymmetry():
    from scripts.dba_kronecker_precision import KroneckerPrecisionFactor

    component = np.array([[2.0, 0.4 + 5e-11], [0.4 - 5e-11, 1.5]])
    prior = np.array([[1.7, 0.35], [0.35, 0.9]])
    information = np.array([0.2, -0.1, 0.4, 0.7])
    symmetric = 0.5 * (component + component.T)
    dense = np.kron(symmetric, np.eye(2)) + np.kron(np.eye(2), prior)

    factor = KroneckerPrecisionFactor.build(component, prior)

    assert factor.solve(information) == pytest.approx(
        np.linalg.solve(dense, information), rel=1e-11, abs=1e-11
    )


def test_factor_rejects_nonsymmetric_or_nonpositive_inputs():
    from scripts.dba_kronecker_precision import KroneckerPrecisionFactor

    with pytest.raises(ValueError, match="component precision"):
        KroneckerPrecisionFactor.build(
            np.array([[1.0, 2.0], [0.0, 1.0]]), np.eye(2)
        )
    with pytest.raises(ValueError, match="prior precision"):
        KroneckerPrecisionFactor.build(np.eye(2), -np.eye(2))


def test_factor_rejects_invalid_information():
    from scripts.dba_kronecker_precision import KroneckerPrecisionFactor

    factor = KroneckerPrecisionFactor.build(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="information"):
        factor.solve(np.ones(3))
    with pytest.raises(ValueError, match="information"):
        factor.solve(np.array([1.0, 2.0, np.nan, 4.0]))
