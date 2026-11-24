"""Residual-likelihood variance components for a fixed linear model.

This is classical error-contrast/REML estimation, not a novel robust estimator.
The contrasts annihilate linear state changes, not nonlinear bias or outliers.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from dranppr_linear_evidence import LinearResidualEvidence


class ContrastVarianceModel:
    def __init__(self, design, measurement, components, base_variance):
        a, y = np.asarray(design, float), np.asarray(measurement, float)
        d, base = np.asarray(components, float), np.asarray(base_variance, float)
        if (a.ndim != 2 or y.shape != (len(a),) or d.ndim != 2 or
                d.shape[0] != len(a) or not d.shape[1] or base.shape != y.shape or
                not all(np.isfinite(x).all() for x in (a, y, d, base)) or
                np.any(d < 0.) or np.any(base < 0.)):
            raise ValueError('invalid variance-component model')
        self.contrasts = LinearResidualEvidence(a).contrast_basis
        if not self.contrasts.shape[1]:
            raise ValueError('no residual degrees of freedom')
        c = self.contrasts
        self.z = c.T @ y
        self.base = c.T @ (base[:, None] * c)
        self.components = np.array([c.T @ (column[:, None] * c) for column in d.T])
        vectors = self.components.reshape(len(d.T), -1).T
        norms = np.linalg.norm(vectors, axis=0)
        self.component_rank = int(np.linalg.matrix_rank(vectors / np.where(norms > 0., norms, 1.)))

    def covariance(self, log_components):
        q = np.exp(log_components)
        return self.base + np.einsum('k,kij->ij', q, self.components)

    def nll(self, log_components):
        covariance = self.covariance(log_components)
        factor = cho_factor(covariance, lower=True, check_finite=False)
        solved = cho_solve(factor, self.z, check_finite=False)
        return float(.5 * (self.z @ solved + 2*np.log(np.diag(factor[0])).sum() +
                           len(self.z)*np.log(2*np.pi)))

    def gradient(self, log_components):
        factor = cho_factor(self.covariance(log_components), lower=True, check_finite=False)
        solved = cho_solve(factor, self.z, check_finite=False)
        return np.exp(log_components) * np.array([
            .5*(np.trace(cho_solve(factor, component, check_finite=False)) -
                solved @ component @ solved) for component in self.components])

    def fisher(self, log_components):
        factor = cho_factor(self.covariance(log_components), lower=True, check_finite=False)
        derivatives = [q*cho_solve(factor, c, check_finite=False)
                       for q, c in zip(np.exp(log_components), self.components)]
        return np.array([[.5*np.trace(a @ b) for b in derivatives] for a in derivatives])

    def fit(self, initial):
        if self.component_rank != len(self.components):
            raise ValueError('variance components are not structurally identifiable')
        result = minimize(self.nll, initial, jac=self.gradient, method='L-BFGS-B',
                          bounds=[(-16., 10.)]*len(self.components),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7})
        information = self.fisher(result.x)
        diagonal = np.sqrt(np.diag(information))
        correlation = information / (diagonal[:, None]*diagonal[None, :])
        return dict(components=np.exp(result.x).tolist(), nll=float(result.fun),
                    success=bool(result.success), message=str(result.message),
                    log_score=self.gradient(result.x).tolist(),
                    at_numerical_boundary=bool(np.any(result.x < -15.99) or np.any(result.x > 9.99)),
                    boundary_components=np.flatnonzero((result.x < -15.99) | (result.x > 9.99)).tolist(),
                    fisher_eigenvalues=np.linalg.eigvalsh(information).tolist(),
                    fisher_correlation=correlation.tolist(),
                    component_rank=self.component_rank, residual_dof=len(self.z),
                    evaluations=int(result.nfev))
