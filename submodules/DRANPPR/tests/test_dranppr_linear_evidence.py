import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_linear_evidence import (
    FactorDeletedBlockEvidence,
    LinearResidualEvidence,
    deleted_group_scale_evidence,
    cross_fitted_group_scale_evidence,
    estimate_group_measurement_scale,
    factor_deleted_scores,
    factor_excluded_prediction,
    profile_deleted_group_scale,
)


def test_decomposition_separates_unfinished_optimization_from_contrast():
    a = np.ones((4, 1))
    evidence = LinearResidualEvidence(a)
    residual = np.array([-1., 1., -1., 1.]) + 3.
    result = evidence.evaluate(residual)
    assert result.rank == 1 and result.residual_dof == 3
    assert result.total_energy == pytest.approx(40.)
    assert result.optimization_energy == pytest.approx(36.)
    assert result.contrast_energy == pytest.approx(4.)
    assert result.scale_squared == pytest.approx(4. / 3.)


def test_contrast_is_invariant_to_any_linear_state_error():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(30, 8))
    residual = rng.normal(size=30)
    evidence = LinearResidualEvidence(a)
    original = evidence.evaluate(residual)
    for scale in (0., .1, 1., 100.):
        moved = evidence.evaluate(residual + a @ (scale * rng.normal(size=8)))
        assert moved.contrast_energy == pytest.approx(original.contrast_energy, abs=1e-9)


def test_handles_gauge_rank_deficiency_and_matches_lstsq():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 4))
    a = np.column_stack([a, a[:, 0] + a[:, 2]])
    residual = rng.normal(size=20)
    result = LinearResidualEvidence(a).evaluate(residual)
    step = np.linalg.lstsq(a, -residual, rcond=None)[0]
    assert result.rank == 4
    assert result.contrast_energy == pytest.approx(np.linalg.norm(residual + a @ step)**2)
    assert result.total_energy == pytest.approx(result.optimization_energy + result.contrast_energy)


def test_no_redundancy_means_no_noise_estimate_not_zero_noise():
    result = LinearResidualEvidence(np.eye(3)).evaluate(np.array([1., 2., 3.]))
    assert result.residual_dof == 0
    assert result.scale_squared is None


def test_observationally_equivalent_bias_is_not_identifiable():
    a = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    evidence = LinearResidualEvidence(a)
    assert evidence.evaluate(a @ np.array([20., -8.])).contrast_energy < 1e-24
    assert evidence.evaluate(np.array([20., 0., 0., 0.])).contrast_energy == pytest.approx(200.)


def test_scale_estimator_has_correct_degrees_of_freedom():
    rng = np.random.default_rng(7)
    evidence = LinearResidualEvidence(rng.normal(size=(40, 15)))
    draws = [evidence.evaluate(2. * rng.normal(size=40)).scale_squared for _ in range(2000)]
    assert np.mean(draws) == pytest.approx(4., rel=.025)


@pytest.mark.parametrize('a', [np.array([[np.nan]]), np.zeros((0, 2)), np.ones(3)])
def test_invalid_design_is_rejected(a):
    with pytest.raises(ValueError):
        LinearResidualEvidence(a)


def test_invalid_residual_is_rejected():
    evidence = LinearResidualEvidence(np.ones((3, 1)))
    for residual in (np.ones(2), np.array([0., np.inf, 1.]), np.ones((3, 1))):
        with pytest.raises(ValueError):
            evidence.evaluate(residual)


def test_factor_excluded_prediction_includes_state_uncertainty():
    a = np.ones((3, 1))
    y = np.array([0., 0., 1.])
    result = factor_excluded_prediction(a, y, [2])
    assert result.identifiable
    np.testing.assert_allclose(result.innovation, [1.])
    np.testing.assert_allclose(result.predictive_covariance, [[1.5]])
    assert result.normalized_energy == pytest.approx(2./3.)


def test_factor_exclusion_blocks_self_validation_of_local_corruption():
    a = np.ones((4, 1))
    clean = factor_excluded_prediction(a, np.zeros(4), [3])
    corrupt = factor_excluded_prediction(a, np.array([0., 0., 0., 10.]), [3])
    assert clean.normalized_energy == pytest.approx(0.)
    assert corrupt.normalized_energy == pytest.approx(75.)


def test_common_state_equivalent_bias_is_not_detected_as_bad_relation():
    a = np.ones((4, 1))
    result = factor_excluded_prediction(a, np.full(4, 5.), [3])
    assert result.identifiable
    assert result.normalized_energy == pytest.approx(0., abs=1e-12)


def test_unobservable_excluded_direction_returns_ambiguity():
    result = factor_excluded_prediction(np.eye(2), np.array([1., 20.]), [1])
    assert not result.identifiable
    assert result.normalized_energy is None
    assert result.unobservable_norm == pytest.approx(1.)


@pytest.mark.parametrize('heldout', [[], [2], [0, 0], [-1]])
def test_invalid_factor_exclusion_is_rejected(heldout):
    with pytest.raises(ValueError):
        factor_excluded_prediction(np.ones((2, 1)), np.zeros(2), heldout)


def test_group_scale_subtracts_predictive_state_uncertainty():
    rng = np.random.default_rng(11)
    count, dimension, true_scale = 10000, 2, 1.
    state = np.repeat((9. * np.eye(dimension))[None], count, axis=0)
    base = np.repeat(np.eye(dimension)[None], count, axis=0)
    innovation = np.sqrt(10.) * rng.normal(size=(count, dimension))
    estimate = estimate_group_measurement_scale(innovation, state, base)
    naive = np.mean(np.sum(innovation**2, axis=1)) / dimension
    assert naive == pytest.approx(10., rel=.03)
    assert estimate.scale == pytest.approx(true_scale, abs=.25)
    assert estimate.sample_count == count


def test_group_scale_handles_varying_anisotropic_predictive_covariance():
    rng = np.random.default_rng(12)
    count, true_scale = 6000, 2.5
    base_one = np.diag([1., 4.])
    base = np.repeat(base_one[None], count, axis=0)
    state = np.array([np.diag([.1 + i/count, 2. - i/count])
                      for i in range(count)])
    innovation = np.array([rng.multivariate_normal(np.zeros(2),
                                                    state[i] + true_scale * base[i])
                           for i in range(count)])
    estimate = estimate_group_measurement_scale(innovation, state, base)
    assert estimate.scale == pytest.approx(true_scale, abs=.12)


def test_group_scale_is_not_a_replacement_for_relation_inference():
    innovation = np.zeros((100, 1))
    innovation[:, 0] = np.random.default_rng(13).normal(size=100)
    innovation[0, 0] = 100.
    covariance = np.zeros((100, 1, 1))
    base = np.ones((100, 1, 1))
    estimate = estimate_group_measurement_scale(innovation, covariance, base)
    assert estimate.scale > 50.


def test_cross_fitted_scale_blocks_covariance_self_validation():
    rng = np.random.default_rng(14)
    innovation = rng.normal(size=(100, 1))
    innovation[0, 0] = 100.
    state = np.zeros((100, 1, 1))
    base = np.ones((100, 1, 1))
    inclusive = estimate_group_measurement_scale(innovation, state, base)
    crossfit = cross_fitted_group_scale_evidence(innovation, state, base)
    assert inclusive.scale > 50.
    assert crossfit.excluded_scales[0] == pytest.approx(1., abs=.25)
    assert crossfit.normalized_energy[0] > 5000.
    assert crossfit.normalized_energy[0] == max(crossfit.normalized_energy)


def test_cross_fitted_scale_still_learns_group_wide_change():
    rng = np.random.default_rng(16)
    innovation = 3. * rng.normal(size=(300, 2))
    state = np.zeros((300, 2, 2))
    base = np.repeat(np.eye(2)[None], 300, axis=0)
    crossfit = cross_fitted_group_scale_evidence(innovation, state, base)
    assert np.median(crossfit.excluded_scales) == pytest.approx(9., abs=1.)
    assert np.median(crossfit.normalized_energy / 2.) == pytest.approx(.7, abs=.2)


def test_cross_fitted_scale_requires_independent_peer():
    with pytest.raises(ValueError):
        cross_fitted_group_scale_evidence(
            np.ones((1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)))


def test_group_scale_rejects_invalid_covariance():
    with pytest.raises(ValueError):
        estimate_group_measurement_scale(
            np.ones((2, 1)), np.zeros((2, 1, 1)), -np.ones((2, 1, 1)))


def test_deleted_block_scores_match_explicit_factor_exclusion():
    rng = np.random.default_rng(15)
    a = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    state = np.linalg.lstsq(a, y, rcond=None)[0]
    residual = a @ state - y
    blocks = [2, 3, 4, 5, 6]
    scores = factor_deleted_scores(a, residual, blocks)
    offset = 0
    for size, score in zip(blocks, scores):
        direct = factor_excluded_prediction(a, y, range(offset, offset + size))
        assert score.identifiable == direct.identifiable
        assert score.normalized_energy == pytest.approx(direct.normalized_energy,
                                                        rel=1e-9, abs=1e-9)
        np.testing.assert_allclose(score.innovation, direct.innovation,
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(score.predictive_covariance,
                                   direct.predictive_covariance,
                                   rtol=1e-9, atol=1e-9)
        offset += size


def test_deleted_block_reports_essential_factor_as_unidentifiable():
    a = np.eye(2)
    scores = factor_deleted_scores(a, np.zeros(2), [1, 1])
    assert not scores[0].identifiable and not scores[1].identifiable
    assert scores[0].normalized_energy is None


def test_deleted_group_scale_uses_predictive_innovations_and_uncertainty():
    rng = np.random.default_rng(19)
    blocks = []
    for innovation in 3. * rng.normal(size=(200, 2)):
        blocks.append(FactorDeletedBlockEvidence(
            row_offset=0, dimension=2, innovation=innovation,
            predictive_covariance=np.eye(2), normalized_energy=float(
                innovation @ innovation), energy_per_dimension=float(
                innovation @ innovation / 2.), identifiable=True,
            maximum_leverage=0., minimum_deleted_precision=1.))

    result = deleted_group_scale_evidence(blocks)

    assert result.inclusive.scale == pytest.approx(9., abs=1.5)
    assert np.median(result.cross_fitted.excluded_scales) == pytest.approx(
        9., abs=1.5)


def test_profiled_deleted_group_scale_recomputes_state_covariance():
    rng = np.random.default_rng(27)
    dimension = 5
    truth = rng.normal(size=dimension)
    reference = rng.normal(size=(50, dimension))
    sensor = rng.normal(size=(200, dimension))
    design = np.vstack([reference, sensor])
    measurement = np.concatenate([
        reference @ truth + rng.normal(size=len(reference)),
        sensor @ truth + 3. * rng.normal(size=len(sensor)),
    ])
    block_sizes = [1] * len(reference) + [2] * (len(sensor) // 2)
    target_group = [False] * len(reference) + [True] * (len(sensor) // 2)

    estimate = profile_deleted_group_scale(
        design, measurement, block_sizes, target_group)
    shifted = profile_deleted_group_scale(
        design, measurement + design @ np.ones(dimension),
        block_sizes, target_group)

    assert estimate.scale == pytest.approx(9., abs=2.)
    assert shifted.scale == pytest.approx(estimate.scale, rel=1e-7)


def test_profiled_group_scale_skips_unidentifiable_heldout_blocks():
    design = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    measurement = np.array([0., 2., 1., -1.])

    estimate = profile_deleted_group_scale(
        design, measurement, [1, 1, 1, 1], [False, True, True, True])

    assert estimate.target_factors == 2
    assert np.isfinite(estimate.scale)


@pytest.mark.parametrize('blocks', [[], [1], [1, 0, 2], [1, 3]])
def test_deleted_block_rejects_invalid_partition(blocks):
    with pytest.raises(ValueError):
        factor_deleted_scores(np.ones((3, 1)), np.zeros(3), blocks)
