"""Exact fixed-linearization leave-block-out noise fitting, an audit reference.

The state response is recomputed for every covariance proposal. This is not a
distributed implementation or a nonlinear conditional-independence guarantee.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.optimize import minimize


class DeletedLinearLikelihood:
    def __init__(self, design, measurement, block_sizes, scored_blocks):
        a = np.asarray(design, dtype=float)
        y = np.asarray(measurement, dtype=float)
        sizes = tuple(block_sizes)
        selected = tuple(scored_blocks)
        if (a.ndim != 2 or not len(a) or y.shape != (len(a),) or
                not np.isfinite(a).all() or not np.isfinite(y).all() or
                not sizes or any(not isinstance(n, (int, np.integer)) or n <= 0
                                 for n in sizes) or sum(sizes) != len(a) or
                not selected or len(set(selected)) != len(selected) or
                any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(sizes)
                    for i in selected)):
            raise ValueError('invalid deleted linear likelihood')
        u, s, _ = svd(a, full_matrices=False, check_finite=False)
        tolerance = max(a.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.)
        self.rank = int(np.sum(s > tolerance))
        self.basis = u[:, :self.rank]
        self.y = y
        offsets = np.cumsum([0, *sizes])
        self.slices = tuple(slice(offsets[i], offsets[i + 1]) for i in range(len(sizes)))
        good, unknown = [], []
        for i in selected:
            basis = self.basis[self.slices[i]]
            precision = np.eye(sizes[i]) - basis @ basis.T
            (good if np.min(np.linalg.eigvalsh(precision)) > 1e-10 else unknown).append(i)
        self.scored_blocks = tuple(good)
        self.unobservable_blocks = tuple(unknown)
        if not good:
            raise ValueError('no identifiable held-out block')

    def predictions(self, variance):
        variance = np.asarray(variance, dtype=float)
        if (variance.shape != self.y.shape or not np.isfinite(variance).all() or
                np.any(variance <= 0.)):
            raise ValueError('variance must be finite and positive')
        scale = np.sqrt(variance)
        basis = self.basis / scale[:, None]
        values = self.y / scale
        if self.rank:
            factor = cho_factor(basis.T @ basis, check_finite=False)
            response = cho_solve(factor, basis.T, check_finite=False)
            residual = values - basis @ (response @ values)
        else:
            response = np.empty((0, len(values)))
            residual = values
        output = []
        for index in self.scored_blocks:
            rows = self.slices[index]
            precision = np.eye(len(values[rows])) - basis[rows] @ response[:, rows]
            precision = .5 * (precision + precision.T)
            factor = cho_factor(precision, check_finite=False)
            covariance_white = cho_solve(factor, np.eye(len(values[rows])), check_finite=False)
            innovation = scale[rows] * (covariance_white @ residual[rows])
            covariance = scale[rows, None] * covariance_white * scale[None, rows]
            output.append((index, innovation, covariance))
        return output

    def nll(self, variance):
        total = 0.
        for _, innovation, covariance in self.predictions(variance):
            chol = np.linalg.cholesky(covariance)
            whitened = np.linalg.solve(chol, innovation)
            total += .5 * (2 * np.sum(np.log(np.diag(chol))) +
                           whitened @ whitened + len(innovation) * np.log(2 * np.pi))
        return float(total)


def fit_polar_noise(design, measurement, block_sizes, observation_blocks,
                    distances, bearing_sigma=.05, range_sigma=.1,
                    conditional=True):
    """Fit bearing variance and range a+b*d^2 using nominal-whitened rows.

    Distances are fixed independent predictions, not the measured ranges. Bounds
    protect numerical factorization only and their activation is reported.
    """
    distances = np.asarray(distances, dtype=float)
    blocks = tuple(observation_blocks)
    if (distances.shape != (len(blocks),) or not np.isfinite(distances).all() or
            np.any(distances <= 0.) or not np.isfinite([bearing_sigma, range_sigma]).all() or
            bearing_sigma <= 0. or range_sigma <= 0. or
            any(block_sizes[i] != 2 for i in blocks)):
        raise ValueError('invalid polar-noise inputs')
    model = DeletedLinearLikelihood(design, measurement, block_sizes, blocks)
    if len(model.scored_blocks) < 2:
        raise ValueError('fewer than two identifiable observations')
    bearing_rows = np.array([model.slices[i].start for i in blocks], dtype=int)
    range_rows = bearing_rows + 1
    distance_scale = float(np.median(distances))
    normalized = (distances / distance_scale)**2

    def variances(parameters):
        q = np.exp(parameters)
        variance = np.ones(len(measurement))
        variance[bearing_rows] = q[0]
        variance[range_rows] = q[1] + (q[2] * normalized if conditional else 0.)
        return variance

    def objective(parameters):
        try:
            return model.nll(variances(parameters)) / len(model.scored_blocks)
        except np.linalg.LinAlgError:
            return float('inf')

    initial = np.log([1., .5, .5] if conditional else [1., 1.])
    result = minimize(objective, initial, method='Powell',
                      bounds=[(-16., 10.)] * len(initial),
                      options={'maxiter': 150, 'ftol': 1e-8, 'xtol': 1e-5})
    q = np.exp(result.x)
    return dict(
        bearing_variance=float(bearing_sigma**2 * q[0]),
        range_intercept_variance=float(range_sigma**2 * q[1]),
        range_distance_squared_coefficient=(
            float(range_sigma**2 * q[2] / distance_scale**2) if conditional else 0.),
        mean_deleted_nll=float(result.fun),
        nominal_mean_deleted_nll=model.nll(np.ones(len(measurement))) / len(model.scored_blocks),
        success=bool(result.success and np.isfinite(result.fun)),
        optimizer_message=str(result.message),
        optimizer='Powell',
        at_numerical_boundary=bool(np.any(result.x < -15.99) or np.any(result.x > 9.99)),
        function_evaluations=int(result.nfev),
        scored_observations=len(model.scored_blocks),
        unobservable_observations=len(model.unobservable_blocks),
        conditional=conditional,
        fixed_linearization=True)
