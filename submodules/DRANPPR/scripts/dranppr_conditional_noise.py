"""Small predictive noise models for DRANPPR evidence allocation.

The fitted variance is measurement variance, not total residual variance:
state predictive variance is supplied separately and is never learned as sensor
noise.  This module is a statistical reference, not a distributed solver.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass(frozen=True)
class ConstantVarianceEstimate:
    variance: float
    mean_nll: float
    success: bool


@dataclass(frozen=True)
class RangeConditionalVarianceEstimate:
    intercept_variance: float
    distance_squared_coefficient: float
    mean_nll: float
    success: bool

    def measurement_variance(self, distance):
        distance = np.asarray(distance, dtype=float)
        return self.intercept_variance + self.distance_squared_coefficient * distance**2


def _inputs(innovation, state_variance, distance=None):
    innovation = np.asarray(innovation, dtype=float)
    state_variance = np.asarray(state_variance, dtype=float)
    if innovation.ndim != 1 or state_variance.shape != innovation.shape or not len(innovation):
        raise ValueError('innovation and state variance must be nonempty vectors of equal size')
    if not np.isfinite(innovation).all() or not np.isfinite(state_variance).all():
        raise ValueError('predictive inputs must be finite')
    if np.any(state_variance < 0):
        raise ValueError('state predictive variance must be nonnegative')
    if distance is None:
        return innovation, state_variance
    distance = np.asarray(distance, dtype=float)
    if distance.shape != innovation.shape or not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('distance must be a finite nonnegative vector')
    return innovation, state_variance, distance


def gaussian_mean_nll(innovation, total_variance):
    innovation = np.asarray(innovation, dtype=float)
    total_variance = np.asarray(total_variance, dtype=float)
    if (innovation.shape != total_variance.shape or innovation.ndim != 1 or
            not len(innovation) or not np.isfinite(innovation).all() or
            not np.isfinite(total_variance).all() or np.any(total_variance <= 0)):
        raise ValueError('invalid innovation or total variance')
    return float(np.mean(.5 * (innovation**2 / total_variance +
                               np.log(total_variance) + math.log(2. * math.pi))))


def fit_constant_measurement_variance(innovation, state_variance=None):
    innovation = np.asarray(innovation, dtype=float)
    if state_variance is None:
        state_variance = np.zeros_like(innovation)
    innovation, state_variance = _inputs(innovation, state_variance)
    reference = max(float(np.mean(innovation**2)),
                    float(np.mean(state_variance)), np.finfo(float).tiny)

    def objective(log_ratio):
        variance = reference * np.exp(float(log_ratio))
        return gaussian_mean_nll(innovation, state_variance + variance)

    result = minimize_scalar(objective, bounds=(-30., 30.), method='bounded')
    variance = reference * np.exp(float(result.x))
    return ConstantVarianceEstimate(
        variance=variance, mean_nll=objective(result.x), success=bool(result.success))


def fit_range_conditional_variance(innovation, distance, state_variance=None):
    innovation = np.asarray(innovation, dtype=float)
    if state_variance is None:
        state_variance = np.zeros_like(innovation)
    innovation, state_variance, distance = _inputs(
        innovation, state_variance, distance)
    constant = fit_constant_measurement_variance(innovation, state_variance)
    distance_scale = max(float(np.median(distance[distance > 0]))
                         if np.any(distance > 0) else 1., np.finfo(float).eps)
    normalized_distance2 = (distance / distance_scale)**2
    # Both optimized variables are dimensionless fractions of the constant
    # measurement variance. Bounds prevent floating-point overflow only.
    initial = np.log([.5, .5])

    def terms(parameters):
        intercept_ratio, slope_ratio = np.exp(parameters)
        measurement = constant.variance * (
            intercept_ratio + slope_ratio * normalized_distance2)
        total = state_variance + measurement
        return intercept_ratio, slope_ratio, measurement, total

    def objective(parameters):
        return gaussian_mean_nll(innovation, terms(parameters)[3])

    def gradient(parameters):
        intercept_ratio, slope_ratio, _, total = terms(parameters)
        derivative = .5 * (1. / total - innovation**2 / total**2)
        common = constant.variance
        return np.array([
            np.mean(derivative * common * intercept_ratio),
            np.mean(derivative * common * slope_ratio * normalized_distance2),
        ])

    result = minimize(objective, initial, jac=gradient, method='L-BFGS-B',
                      bounds=[(-30., 30.), (-30., 30.)])
    intercept_ratio, slope_ratio, _, _ = terms(result.x)
    return RangeConditionalVarianceEstimate(
        intercept_variance=constant.variance * intercept_ratio,
        distance_squared_coefficient=(constant.variance * slope_ratio /
                                      distance_scale**2),
        mean_nll=objective(result.x), success=bool(result.success))
