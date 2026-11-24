"""Exact diagonal-variance REML using sparse square-root state elimination.

Same likelihood as orthonormal error contrasts for a full-rank, gauge-fixed
design. No stochastic traces, variance priors, damping or automatic gauge fix.
Sparse QR avoids explicitly squaring the state-design condition number.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
import sparseqr


class SparseVarianceModel:
    def __init__(self, design, measurement, components, base_variance, chunk_size=64):
        a = sparse.csc_matrix(design, dtype=float)
        y, d, base = (np.asarray(x, float) for x in (measurement, components, base_variance))
        m, n = a.shape
        if (not 0 < n < m or y.shape != (m,) or d.ndim != 2 or d.shape[0] != m or
                not d.shape[1] or base.shape != (m,) or np.any(d < 0.) or np.any(base < 0.) or
                not all(np.isfinite(x).all() for x in (a.data, y, d, base)) or
                type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError('invalid sparse variance model or no residual degrees of freedom')
        scale = np.sqrt(np.asarray(a.multiply(a).sum(axis=0)).ravel())
        if np.any(scale <= 0.):
            raise ValueError('zero column in rank-deficient design')
        self.a = (a @ sparse.diags(1./scale)).tocsr()
        self.y, self.d, self.base = y, d, base
        self.chunk_size, self.residual_dof = chunk_size, m-n
        self.max_solve_residual = 0.
        self._theta = None
        _, _, _, self.reference_logdet = self._qr(self.a, np.zeros(m))

    @staticmethod
    def _qr(a, rhs):
        z, r, permutation, rank = sparseqr.rz(a.tocoo(), rhs, tolerance=-2.)
        pivots = np.abs(r.diagonal())
        if rank != a.shape[1] or not np.isfinite(pivots).all() or np.any(pivots <= 0.):
            raise ValueError('numerically rank-deficient state design')
        if permutation is None:
            permutation = np.arange(a.shape[1])
        return r.tocsc(), np.asarray(z).ravel(), permutation, float(2*np.log(pivots).sum())

    def _solve_triangular(self, rhs, transpose=False):
        solution = self._lu.solve(rhs, trans='T' if transpose else 'N')
        residual = np.linalg.norm((self._r.T if transpose else self._r) @ solution-rhs)/max(1., np.linalg.norm(rhs))
        self.max_solve_residual = max(self.max_solve_residual, float(residual))
        if not np.isfinite(residual) or residual > 1e-8:
            raise RuntimeError('sparse triangular solve residual too large')
        return solution

    def _prepare(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.d.shape[1],) or not np.isfinite(theta).all():
            raise ValueError('invalid log variances')
        if self._theta is not None and np.array_equal(self._theta, theta):
            return
        q = np.exp(theta)
        variance = self.base + self.d @ q
        if not np.isfinite(variance).all() or np.any(variance <= 0.):
            raise ValueError('nonpositive observation variance')
        self._w = 1./variance
        self._r, projected, self._permutation, logdet = self._qr(
            self.a.multiply(np.sqrt(self._w)[:, None]), np.sqrt(self._w)*self.y)
        self._lu = splu(self._r, permc_spec='NATURAL', diag_pivot_thresh=0.)
        state = np.empty(self.a.shape[1])
        state[self._permutation] = self._solve_triangular(projected)
        residual = self.y-self.a @ state
        self._alpha = self._w*residual
        self._nll = float(.5*(np.log(variance).sum()+logdet-self.reference_logdet+
                             residual @ self._alpha+self.residual_dof*np.log(2*np.pi)))
        self._q, self._theta = q, theta.copy()
        self._gradient = self._fisher = None

    def nll(self, theta):
        self._prepare(theta)
        return self._nll

    def _information(self, fisher):
        m = self.a.shape[0]
        diagonal = np.empty(m)
        information = np.zeros((self.d.shape[1], self.d.shape[1]))
        derivatives = self.d*self._q
        for start in range(0, m, self.chunk_size):
            stop = min(start+self.chunk_size, m)
            block = self.a[start:stop]
            projected = self._solve_triangular(block[:, self._permutation].T.toarray(), transpose=True)
            leverage = np.sum(projected**2, axis=0)
            diagonal[start:stop] = self._w[start:stop]-self._w[start:stop]**2*leverage
            if fisher:
                solved = np.empty(projected.shape)
                solved[self._permutation] = self._solve_triangular(projected)
                # Exact columns of P; only this temporary block is dense.
                columns = -(self.a @ solved)*self._w[:, None]*self._w[None, start:stop]
                columns[np.arange(start, stop), np.arange(stop-start)] += self._w[start:stop]
                information += .5*(derivatives.T @ (columns*columns)) @ derivatives[start:stop]
        self._gradient = .5*self._q*(self.d.T @ (diagonal-self._alpha**2))
        if fisher:
            self._fisher = .5*(information+information.T)

    def gradient(self, theta):
        self._prepare(theta)
        if self._gradient is None:
            self._information(False)
        return self._gradient.copy()

    def fisher(self, theta):
        self._prepare(theta)
        if self._fisher is None:
            self._information(True)
        return self._fisher.copy()

    def fit(self, initial):
        information = self.fisher(np.zeros(self.d.shape[1]))
        norm = np.sqrt(np.diag(information))
        scaled = information / np.maximum(norm[:, None]*norm[None, :], np.finfo(float).tiny)
        component_rank = int(np.linalg.matrix_rank(scaled))
        if component_rank != self.d.shape[1]:
            raise ValueError('variance components are not structurally identifiable')
        result = minimize(self.nll, initial, jac=self.gradient, method='L-BFGS-B',
                          bounds=[(-16., 10.)]*self.d.shape[1],
                          options={'maxiter':200, 'ftol':1e-12, 'gtol':1e-7})
        information = self.fisher(result.x)
        diagonal = np.sqrt(np.diag(information))
        return dict(components=np.exp(result.x).tolist(), nll=float(result.fun),
            success=bool(result.success), message=str(result.message),
            log_score=self.gradient(result.x).tolist(),
            at_numerical_boundary=bool(np.any(result.x < -15.99) or np.any(result.x > 9.99)),
            boundary_components=np.flatnonzero((result.x < -15.99) | (result.x > 9.99)).tolist(),
            fisher_eigenvalues=np.linalg.eigvalsh(information).tolist(),
            fisher_correlation=(information/(diagonal[:, None]*diagonal[None, :])).tolist(),
            component_rank=component_rank, residual_dof=self.residual_dof,
            evaluations=int(result.nfev), max_solve_residual=self.max_solve_residual,
            implementation='sparse_square_root_reml_exact_blocked_traces')
