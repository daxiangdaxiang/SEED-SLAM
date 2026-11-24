"""Exact linear algebra for DBA's two-coordinate canonical precision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, cho_solve


@dataclass(frozen=True)
class KroneckerPrecisionFactor:
    """Factor K (x) I_2 + I (x) P through the eigensystem of P."""

    eigenvectors: np.ndarray
    factors: tuple[np.ndarray, np.ndarray]
    half_log_determinant: float

    @classmethod
    def build(
        cls,
        component_precision: np.ndarray,
        prior_precision: np.ndarray,
    ) -> "KroneckerPrecisionFactor":
        component = np.asarray(component_precision, dtype=float)
        prior = np.asarray(prior_precision, dtype=float)
        if (
            component.ndim != 2
            or component.shape[0] != component.shape[1]
            or not np.all(np.isfinite(component))
            or not np.allclose(
                component, component.T, atol=1e-9, rtol=1e-10
            )
        ):
            raise ValueError(
                "component precision must be finite, square, and symmetric"
            )
        component = 0.5 * (component + component.T)
        if (
            prior.shape != (2, 2)
            or not np.all(np.isfinite(prior))
            or not np.allclose(prior, prior.T, atol=1e-12, rtol=1e-12)
        ):
            raise ValueError("prior precision must be finite symmetric 2x2")
        eigenvalues, eigenvectors = np.linalg.eigh(prior)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("prior precision must be positive definite")
        if component.shape[0] == 0:
            empty = np.empty((0, 0), dtype=float)
            return cls(
                np.array(eigenvectors, copy=True),
                (np.array(empty, copy=True), np.array(empty, copy=True)),
                0.0,
            )
        identity = np.eye(component.shape[0])
        try:
            factors = tuple(
                cholesky(
                    component + eigenvalue * identity,
                    lower=True,
                    check_finite=False,
                )
                for eigenvalue in eigenvalues
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "joint canonical precision must be positive definite"
            ) from error
        half_logdet = float(
            sum(np.log(np.diag(factor)).sum() for factor in factors)
        )
        return cls(
            np.array(eigenvectors, copy=True),
            (
                np.array(factors[0], copy=True),
                np.array(factors[1], copy=True),
            ),
            half_logdet,
        )

    def solve(self, information: np.ndarray) -> np.ndarray:
        """Apply the inverse precision to component-major 2D information."""
        values = np.asarray(information, dtype=float)
        component_count = self.factors[0].shape[0]
        if values.shape != (2 * component_count,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("information has incompatible shape or values")
        if component_count == 0:
            return np.array(values, copy=True)
        rotated = values.reshape(component_count, 2) @ self.eigenvectors
        solved = np.empty_like(rotated)
        for column, factor in enumerate(self.factors):
            solved[:, column] = cho_solve(
                (factor, True), rotated[:, column], check_finite=False
            )
        return (solved @ self.eigenvectors.T).reshape(-1)
