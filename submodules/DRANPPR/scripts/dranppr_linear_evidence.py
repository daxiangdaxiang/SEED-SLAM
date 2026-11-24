"""Fixed-linearization residual diagnostic, not a fault classifier."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ResidualDecomposition:
    rank: int
    residual_dof: int
    total_energy: float
    optimization_energy: float
    contrast_energy: float
    scale_squared: Optional[float]


@dataclass(frozen=True)
class FactorExcludedPrediction:
    innovation: np.ndarray
    predictive_covariance: np.ndarray
    normalized_energy: Optional[float]
    identifiable: bool
    training_rank: int
    unobservable_norm: float


@dataclass(frozen=True)
class PredictiveScaleEstimate:
    scale: float
    negative_log_likelihood: float
    sample_count: int
    dimension: int
    at_zero_boundary: bool


@dataclass(frozen=True)
class FactorDeletedBlockEvidence:
    row_offset: int
    dimension: int
    innovation: Optional[np.ndarray]
    predictive_covariance: Optional[np.ndarray]
    normalized_energy: Optional[float]
    energy_per_dimension: Optional[float]
    identifiable: bool
    maximum_leverage: float
    minimum_deleted_precision: float


@dataclass(frozen=True)
class CrossFittedGroupScaleEvidence:
    excluded_scales: np.ndarray
    normalized_energy: np.ndarray
    negative_log_likelihood: np.ndarray


@dataclass(frozen=True)
class DeletedGroupScaleEvidence:
    inclusive: PredictiveScaleEstimate
    cross_fitted: CrossFittedGroupScaleEvidence


@dataclass(frozen=True)
class ProfiledDeletedGroupScale:
    scale: float
    negative_log_likelihood: float
    target_factors: int
    function_evaluations: int


class LinearResidualEvidence:
    """Dense reference for small, already-whitened physical factor graphs.

    Contrasts annihilate changes in the column space of this fixed Jacobian.
    They do not remove nonlinear linearization error, outliers, or model bias.
    The scale estimate assumes one unknown multiplier of a known covariance
    shape and independent zero-mean Gaussian errors. No robust weights or AL
    penalty factors belong in the input used to test that assumption.
    """

    def __init__(self, jacobian):
        a = np.asarray(jacobian, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0 or not np.isfinite(a).all():
            raise ValueError('expected a finite nonempty Jacobian matrix')
        u, singular, _ = svd(a, full_matrices=True)
        tolerance = max(a.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
        self.rank = int(np.sum(singular > tolerance))
        self.rows = a.shape[0]
        self.fit_basis = u[:, :self.rank]
        self.contrast_basis = u[:, self.rank:]

    def evaluate(self, residual):
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.rows,) or not np.isfinite(residual).all():
            raise ValueError('residual must match Jacobian rows and be finite')
        reducible = self.fit_basis.T @ residual
        contrasts = self.contrast_basis.T @ residual
        dof = self.rows - self.rank
        contrast_energy = float(contrasts @ contrasts)
        return ResidualDecomposition(
            self.rank, dof, float(residual @ residual), float(reducible @ reducible),
            contrast_energy, contrast_energy / dof if dof else None)


def factor_excluded_prediction(jacobian, measurement, heldout):
    """Exact linear-Gaussian reference with unit measurement covariance.

    The excluded rows cannot shape their own state prediction. If training rows
    do not observe a direction needed by an excluded row, the evidence is
    explicitly unidentifiable instead of assigning it a finite surprise score.
    """
    a = np.asarray(jacobian, dtype=float)
    y = np.asarray(measurement, dtype=float)
    heldout = np.asarray(heldout)
    if (a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all() or
            y.shape != (a.shape[0],) or not np.isfinite(y).all() or
            heldout.ndim != 1 or len(heldout) == 0 or
            not np.issubdtype(heldout.dtype, np.integer) or
            np.any(heldout < 0) or np.any(heldout >= a.shape[0]) or
            len(np.unique(heldout)) != len(heldout) or len(heldout) == a.shape[0]):
        raise ValueError('invalid linear prediction problem or heldout rows')
    train_mask = np.ones(a.shape[0], dtype=bool)
    train_mask[heldout] = False
    train, train_y = a[train_mask], y[train_mask]
    query, query_y = a[heldout], y[heldout]
    u, singular, vt = svd(train, full_matrices=False)
    tolerance = max(train.shape) * np.finfo(float).eps * (
        singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    basis = vt[:rank].T
    if rank:
        state = basis @ ((u[:, :rank].T @ train_y) / singular[:rank])
        state_covariance = (basis / singular[:rank]) @ (basis / singular[:rank]).T
        row_projection = query @ basis @ basis.T
    else:
        state = np.zeros(a.shape[1])
        state_covariance = np.zeros((a.shape[1], a.shape[1]))
        row_projection = np.zeros_like(query)
    unobservable_norm = float(np.linalg.norm(query - row_projection))
    observable_tolerance = 1e-10 * max(1., float(np.linalg.norm(query)))
    identifiable = unobservable_norm <= observable_tolerance
    innovation = query_y - query @ state
    predictive_covariance = (np.eye(len(heldout)) +
                             query @ state_covariance @ query.T)
    normalized_energy = None
    if identifiable:
        normalized_energy = float(innovation @ np.linalg.solve(
            predictive_covariance, innovation))
    return FactorExcludedPrediction(
        innovation, predictive_covariance, normalized_energy, identifiable,
        rank, unobservable_norm)


def estimate_group_measurement_scale(innovation, state_covariance,
                                     base_measurement_covariance):
    """Profile one sensor-group scale from independent predictive innovations.

    For sample i this minimizes the normalized Gaussian likelihood with
    `S_i(q) = P_i + q R_i`, q >= 0. It is a statistical reference, not a
    relation-robust estimator: one wrong factor can inflate the group scale.
    """
    innovation = np.asarray(innovation, dtype=float)
    state = np.asarray(state_covariance, dtype=float)
    base = np.asarray(base_measurement_covariance, dtype=float)
    if innovation.ndim != 2:
        raise ValueError('innovations must be sample by dimension')
    count, dimension = innovation.shape
    expected = (count, dimension, dimension)
    if (count == 0 or state.shape != expected or base.shape != expected or
            not np.isfinite(innovation).all() or not np.isfinite(state).all() or
            not np.isfinite(base).all() or
            not np.allclose(state, state.transpose(0, 2, 1), atol=1e-12) or
            not np.allclose(base, base.transpose(0, 2, 1), atol=1e-12)):
        raise ValueError('invalid predictive covariance arrays')
    state_eigenvalues = np.linalg.eigvalsh(state)
    base_eigenvalues = np.linalg.eigvalsh(base)
    if np.min(state_eigenvalues) < -1e-12 or np.min(base_eigenvalues) <= 0.:
        raise ValueError('state covariance must be PSD and base covariance SPD')

    def objective(scale):
        covariance = state + scale * base
        sign, logdet = np.linalg.slogdet(covariance)
        if np.any(sign <= 0) or not np.isfinite(logdet).all():
            return float('inf')
        solved = np.linalg.solve(covariance, innovation[..., None])[..., 0]
        return .5 * float(np.sum(logdet + np.sum(innovation * solved, axis=1)))

    whitened = np.linalg.solve(base, innovation[..., None])[..., 0]
    raw_scale = float(np.sum(innovation * whitened) / (count * dimension))
    upper = max(1., 2. * raw_scale)
    for _ in range(64):
        if objective(2. * upper) >= objective(upper):
            break
        upper *= 2.
    else:
        raise RuntimeError('failed to bracket predictive scale likelihood')
    optimum = minimize_scalar(objective, bounds=(0., upper), method='bounded',
                              options={'xatol': 1e-10})
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError('predictive scale optimization failed')
    zero_objective = objective(0.)
    at_zero = np.isfinite(zero_objective) and zero_objective <= optimum.fun
    scale = 0. if at_zero else float(optimum.x)
    nll = zero_objective if at_zero else float(optimum.fun)
    return PredictiveScaleEstimate(scale, nll, count, dimension, at_zero)


def cross_fitted_group_scale_evidence(innovation, state_covariance,
                                      base_measurement_covariance):
    """Leave each factor out of both state prediction and scale calibration.

    Inputs are already factor-excluded predictive innovations/covariances. This
    second exclusion prevents one anomalous relation from enlarging the sensor
    scale used to judge itself. It is a dense audit reference; production code
    should use sufficient-statistic influence updates rather than N refits.
    """
    innovation = np.asarray(innovation, dtype=float)
    state = np.asarray(state_covariance, dtype=float)
    base = np.asarray(base_measurement_covariance, dtype=float)
    estimate_group_measurement_scale(innovation, state, base)
    count, dimension = innovation.shape
    if count < 2:
        raise ValueError('cross-fitted scale needs at least two factors')
    scales, energy, nll = [], [], []
    for heldout in range(count):
        keep = np.arange(count) != heldout
        estimate = estimate_group_measurement_scale(
            innovation[keep], state[keep], base[keep])
        covariance = state[heldout] + estimate.scale * base[heldout]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            raise RuntimeError('cross-fitted predictive covariance is singular')
        solved = np.linalg.solve(covariance, innovation[heldout])
        value = float(innovation[heldout] @ solved)
        scales.append(estimate.scale)
        energy.append(value)
        nll.append(.5 * (logdet + value + dimension * np.log(2. * np.pi)))
    return CrossFittedGroupScaleEvidence(
        np.asarray(scales), np.asarray(energy), np.asarray(nll))


def deleted_group_scale_evidence(blocks):
    """Fit one local group scale from exact deleted-block predictions.

    The factor graph is already whitened by its base covariance. The deleted
    predictive covariance is therefore decomposed as `P_state + q I`, with
    `P_state` frozen at the current `q=1` linearization for this audit.
    """
    blocks = tuple(blocks)
    if len(blocks) < 2:
        raise ValueError('deleted group scale needs at least two factors')
    dimension = blocks[0].dimension
    if (dimension <= 0 or any(
            not block.identifiable or block.dimension != dimension or
            block.innovation is None or block.predictive_covariance is None
            for block in blocks)):
        raise ValueError('deleted group blocks must be identifiable and uniform')
    innovation = np.stack([block.innovation for block in blocks])
    predictive = np.stack([block.predictive_covariance for block in blocks])
    identity = np.eye(dimension)
    state = .5 * (predictive + predictive.transpose(0, 2, 1)) - identity
    base = np.repeat(identity[None], len(blocks), axis=0)
    inclusive = estimate_group_measurement_scale(innovation, state, base)
    cross_fitted = cross_fitted_group_scale_evidence(
        innovation, state, base)
    return DeletedGroupScaleEvidence(inclusive, cross_fitted)


def profile_deleted_group_scale(jacobian, measurement, block_sizes,
                                target_group):
    """Profile one group scale while recomputing the deleted state response."""
    design = np.asarray(jacobian, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    block_sizes = tuple(block_sizes)
    target_group = np.asarray(target_group)
    if (design.ndim != 2 or design.shape[0] == 0 or
            measurement.shape != (design.shape[0],) or
            not np.isfinite(design).all() or
            not np.isfinite(measurement).all() or not block_sizes or
            any(isinstance(size, bool) or
                not isinstance(size, (int, np.integer)) or size <= 0
                for size in block_sizes) or sum(block_sizes) != design.shape[0] or
            target_group.shape != (len(block_sizes),) or
            target_group.dtype != np.bool_ or np.sum(target_group) < 2):
        raise ValueError('invalid profiled deleted-group scale problem')
    row_group = np.repeat(target_group, block_sizes)
    base_state = np.linalg.lstsq(design, measurement, rcond=None)[0]
    base_residual = design @ base_state - measurement
    base_deleted = factor_deleted_scores(design, base_residual, block_sizes)
    scored_group = target_group & np.asarray(
        [evidence.identifiable for evidence in base_deleted], dtype=bool)
    if np.sum(scored_group) < 2:
        raise ValueError('fewer than two identifiable target factors')

    def objective(scale):
        if not np.isfinite(scale) or scale <= 0.:
            return float('inf')
        row_scale = np.where(row_group, np.sqrt(scale), 1.)
        weighted_design = design / row_scale[:, None]
        weighted_measurement = measurement / row_scale
        state = np.linalg.lstsq(
            weighted_design, weighted_measurement, rcond=None)[0]
        residual = weighted_design @ state - weighted_measurement
        deleted = factor_deleted_scores(
            weighted_design, residual, block_sizes)
        total = 0.
        for selected, evidence in zip(scored_group, deleted):
            if not selected:
                continue
            if not evidence.identifiable:
                return float('inf')
            covariance = scale * evidence.predictive_covariance
            innovation = np.sqrt(scale) * evidence.innovation
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0 or not np.isfinite(logdet):
                return float('inf')
            energy = innovation @ np.linalg.solve(covariance, innovation)
            total += .5 * (logdet + energy +
                           evidence.dimension * np.log(2. * np.pi))
        return float(total)

    raw_scale = float(np.mean(base_residual[row_group]**2))
    upper = max(1., 2. * raw_scale)
    for _ in range(32):
        if objective(2. * upper) >= objective(upper):
            break
        upper *= 2.
    else:
        raise RuntimeError('failed to bracket profiled group scale')
    lower = min(1e-6, upper * 1e-8)
    result = minimize_scalar(
        lambda log_scale: objective(np.exp(log_scale)),
        bounds=(np.log(lower), np.log(upper)), method='bounded',
        options={'xatol': 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('profiled deleted-group scale optimization failed')
    return ProfiledDeletedGroupScale(
        float(np.exp(result.x)), float(result.fun), int(np.sum(scored_group)),
        int(result.nfev))


def factor_deleted_scores(jacobian, residual, block_sizes):
    """Compute exact deleted-block predictive scores from one linear fit.

    This is the block PRESS identity. It is intended for audits and for checking
    compact response approximations; forming the dense left singular basis is
    not an online implementation strategy.
    """
    a = np.asarray(jacobian, dtype=float)
    residual = np.asarray(residual, dtype=float)
    block_sizes = tuple(block_sizes)
    if (a.ndim != 2 or a.shape[0] == 0 or not np.isfinite(a).all() or
            residual.shape != (a.shape[0],) or not np.isfinite(residual).all() or
            not block_sizes or any(isinstance(size, bool) or
                                   not isinstance(size, (int, np.integer)) or
                                   size <= 0 for size in block_sizes) or
            sum(block_sizes) != a.shape[0]):
        raise ValueError('invalid Jacobian, residual, or factor block partition')
    u, singular, _ = svd(a, full_matrices=False)
    tolerance = max(a.shape) * np.finfo(float).eps * (
        singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    fit_basis = u[:, :rank]
    result, offset = [], 0
    for size in block_sizes:
        rows = slice(offset, offset + size)
        local_basis = fit_basis[rows]
        leverage = local_basis @ local_basis.T
        deleted_precision = np.eye(size) - leverage
        eigenvalues = np.linalg.eigvalsh(deleted_precision)
        minimum_precision = float(np.min(eigenvalues))
        identifiable = minimum_precision > 1e-10
        innovation = predictive_covariance = energy = None
        if identifiable:
            predictive_covariance = np.linalg.solve(
                deleted_precision, np.eye(size))
            innovation = -predictive_covariance @ residual[rows]
            energy = float(innovation @ deleted_precision @ innovation)
            energy = max(0., energy)
        result.append(FactorDeletedBlockEvidence(
            offset, size, innovation, predictive_covariance, energy,
            energy / size if energy is not None else None, identifiable,
            float(np.max(np.linalg.eigvalsh(leverage))), minimum_precision))
        offset += size
    return result
