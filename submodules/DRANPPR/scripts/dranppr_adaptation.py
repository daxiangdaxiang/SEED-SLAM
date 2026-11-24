"""Small linear-Gaussian reference for cause-separated noise adaptation."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class WeightedStateFit:
    state: np.ndarray
    # Covariance only on the identifiable subspace; use predict for a functional.
    covariance: np.ndarray
    rank: int
    row_basis: np.ndarray
    covariance_root: np.ndarray
    relative_rank_tolerance: float

    def predict(self, rows):
        rows = np.asarray(rows, dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != len(self.state) or
                not np.isfinite(rows).all()):
            raise ValueError('invalid prediction rows')
        if self.rank < len(self.state):
            missing = rows - (rows @ self.row_basis.T) @ self.row_basis
            if np.any(np.linalg.norm(missing, axis=1) >
                      self.relative_rank_tolerance*np.linalg.norm(rows, axis=1)):
                raise UnobservablePredictionError(
                    'prediction is not observable from independent factors')
        projected_root = rows @ self.covariance_root
        return rows @ self.state, np.sum(projected_root**2, axis=1)


class UnobservablePredictionError(ValueError):
    """Insufficient evidence, not a rejected association or a sensor fault."""


@dataclass(frozen=True)
class RobustStateFit:
    state: np.ndarray
    weights: np.ndarray
    iterations: int


def weighted_state_fit(design, measurement, variance):
    design = np.asarray(design, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if (design.ndim != 2 or design.shape[0] == 0 or
            measurement.shape != (design.shape[0],) or
            variance.shape != (design.shape[0],) or
            not np.isfinite(design).all() or
            not np.isfinite(measurement).all() or
            not np.isfinite(variance).all() or np.any(variance <= 0.)):
        raise ValueError('invalid weighted linear state problem')
    whitened_design = design / np.sqrt(variance)[:, None]
    whitened_measurement = measurement / np.sqrt(variance)
    # One square-root rank decision for both the estimate and its uncertainty.
    # Applying a fresh pinv cutoff to A.T @ A can discard weak observable modes.
    left, singular, right = np.linalg.svd(whitened_design, full_matrices=False)
    tolerance = np.finfo(float).eps*max(whitened_design.shape)
    keep = singular > tolerance*singular[0] if singular.size else np.zeros(0, dtype=bool)
    basis = right[keep]
    inverse_root = basis.T/singular[keep]
    state = inverse_root @ (left[:, keep].T @ whitened_measurement)
    covariance = inverse_root @ inverse_root.T
    if not np.isfinite(state).all() or not np.isfinite(covariance).all():
        raise ValueError('nonfinite weighted fit or subspace covariance')
    return WeightedStateFit(state, covariance, int(np.sum(keep)), basis,
                            inverse_root, tolerance)


def huber_state_fit(design, measurement, nominal_variance, delta=1.345,
                    max_iterations=100, tolerance=1e-10):
    """Converged Huber IRLS control under a fixed nominal covariance."""
    design = np.asarray(design, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    nominal_variance = np.asarray(nominal_variance, dtype=float)
    if (not np.isfinite(delta) or delta <= 0. or max_iterations < 1 or
            not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError('invalid Huber control parameters')
    state = weighted_state_fit(
        design, measurement, nominal_variance).state
    weights = np.ones(len(measurement))
    for iteration in range(1, max_iterations + 1):
        standardized = ((measurement - design @ state) /
                        np.sqrt(nominal_variance))
        magnitude = np.abs(standardized)
        weights = np.ones_like(magnitude)
        tail = magnitude > delta
        weights[tail] = delta / magnitude[tail]
        candidate = weighted_state_fit(
            design, measurement, nominal_variance / weights).state
        if np.linalg.norm(candidate - state) <= tolerance * (
                1. + np.linalg.norm(state)):
            state = candidate
            break
        state = candidate
    standardized = ((measurement - design @ state) /
                    np.sqrt(nominal_variance))
    magnitude = np.abs(standardized)
    weights = np.ones_like(magnitude)
    tail = magnitude > delta
    weights[tail] = delta / magnitude[tail]
    return RobustStateFit(state, weights, iteration)


def gnc_geman_mcclure_fit(design, measurement, nominal_variance,
                          cutoff_sigma=2., continuation_factor=1.4,
                          max_inner_iterations=50, tolerance=1e-10):
    """GNC-style Geman--McClure IRLS control, not an official baseline."""
    design = np.asarray(design, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    nominal_variance = np.asarray(nominal_variance, dtype=float)
    if (not np.isfinite(cutoff_sigma) or cutoff_sigma <= 0. or
            not np.isfinite(continuation_factor) or
            continuation_factor <= 1. or max_inner_iterations < 1 or
            not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError('invalid GNC control parameters')
    state = weighted_state_fit(
        design, measurement, nominal_variance).state
    standardized = ((measurement - design @ state) /
                    np.sqrt(nominal_variance))
    cutoff_squared = cutoff_sigma**2
    mu = max(1., float(np.max(standardized**2) / cutoff_squared))
    iterations = 0
    while True:
        for _ in range(max_inner_iterations):
            standardized = ((measurement - design @ state) /
                            np.sqrt(nominal_variance))
            numerator = mu * cutoff_squared
            weights = (numerator /
                       (standardized**2 + numerator))**2
            candidate = weighted_state_fit(
                design, measurement, nominal_variance / weights).state
            iterations += 1
            if np.linalg.norm(candidate - state) <= tolerance * (
                    1. + np.linalg.norm(state)):
                state = candidate
                break
            state = candidate
        if mu == 1.:
            break
        mu = max(1., mu / continuation_factor)
    standardized = ((measurement - design @ state) /
                    np.sqrt(nominal_variance))
    weights = (cutoff_squared /
               (standardized**2 + cutoff_squared))**2
    return RobustStateFit(state, weights, iterations)


def naive_scale_from_state(sensor_design, sensor_measurement, state):
    sensor_design = np.asarray(sensor_design, dtype=float)
    sensor_measurement = np.asarray(sensor_measurement, dtype=float)
    state = np.asarray(state, dtype=float)
    if (sensor_design.ndim != 2 or sensor_design.shape[0] == 0 or
            sensor_measurement.shape != (sensor_design.shape[0],) or
            state.shape != (sensor_design.shape[1],) or
            not np.isfinite(sensor_design).all() or
            not np.isfinite(sensor_measurement).all() or
            not np.isfinite(state).all()):
        raise ValueError('invalid sensor residual problem')
    residual = sensor_measurement - sensor_design @ state
    return float(residual @ residual / len(residual))


def _predictive_objective(reference_design, reference_measurement,
                          sensor_design, sensor_measurement, scale):
    count = len(sensor_measurement)
    total = 0.
    for heldout in range(count):
        keep = np.arange(count) != heldout
        design = np.vstack([reference_design, sensor_design[keep]])
        measurement = np.concatenate([reference_measurement,
                                      sensor_measurement[keep]])
        variance = np.concatenate([np.ones(len(reference_measurement)),
                                   np.full(np.sum(keep), scale)])
        fit = weighted_state_fit(design, measurement, variance)
        row = sensor_design[heldout]
        means, variances = fit.predict(row[None, :])
        predictive_variance = scale + float(variances[0])
        innovation = float(sensor_measurement[heldout] - means[0])
        total += .5 * (np.log(predictive_variance) +
                       innovation**2 / predictive_variance)
    return float(total)


def predictive_group_scale(reference_design, reference_measurement,
                           sensor_design, sensor_measurement):
    """Profile one group variance from leave-factor-out predictions."""
    reference_design = np.asarray(reference_design, dtype=float)
    reference_measurement = np.asarray(reference_measurement, dtype=float)
    sensor_design = np.asarray(sensor_design, dtype=float)
    sensor_measurement = np.asarray(sensor_measurement, dtype=float)
    if (reference_design.ndim != 2 or sensor_design.ndim != 2 or
            reference_design.shape[1] != sensor_design.shape[1] or
            reference_measurement.shape != (reference_design.shape[0],) or
            sensor_measurement.shape != (sensor_design.shape[0],) or
            len(sensor_measurement) < 2 or
            not np.isfinite(reference_design).all() or
            not np.isfinite(reference_measurement).all() or
            not np.isfinite(sensor_design).all() or
            not np.isfinite(sensor_measurement).all()):
        raise ValueError('invalid predictive group-scale problem')
    reference_fit = weighted_state_fit(
        reference_design, reference_measurement,
        np.ones(len(reference_measurement)))
    raw_scale = naive_scale_from_state(
        sensor_design, sensor_measurement, reference_fit.state)
    lower, upper = 1e-6, max(1., 2. * raw_scale)
    objective = lambda scale: _predictive_objective(
        reference_design, reference_measurement,
        sensor_design, sensor_measurement, scale)
    for _ in range(32):
        if objective(2. * upper) >= objective(upper):
            break
        upper *= 2.
    else:
        raise RuntimeError('failed to bracket predictive group scale')
    result = minimize_scalar(objective, bounds=(lower, upper), method='bounded',
                             options={'xatol': 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('predictive group-scale optimization failed')
    return float(result.x)


def candidate_predictive_nll(reference_design, reference_measurement,
                             peer_design, peer_measurement, candidate_rows,
                             measurement, sensor_scale):
    """Score alternative relation rows without assimilating the tested factor."""
    candidate_rows = np.asarray(candidate_rows, dtype=float)
    if (candidate_rows.ndim != 2 or not np.isfinite(candidate_rows).all() or
            not np.isfinite(measurement) or not np.isfinite(sensor_scale) or
            sensor_scale <= 0.):
        raise ValueError('invalid relation candidates or sensor scale')
    design = np.vstack([reference_design, peer_design])
    values = np.concatenate([reference_measurement, peer_measurement])
    variance = np.concatenate([np.ones(len(reference_measurement)),
                               np.full(len(peer_measurement), sensor_scale)])
    fit = weighted_state_fit(design, values, variance)
    if candidate_rows.shape[1] != len(fit.state):
        raise ValueError('candidate relation dimension mismatch')
    means, variances = fit.predict(candidate_rows)
    innovation = measurement - means
    predictive_variance = sensor_scale + variances
    return .5 * (np.log(2. * np.pi * predictive_variance) +
                 innovation**2 / predictive_variance)
