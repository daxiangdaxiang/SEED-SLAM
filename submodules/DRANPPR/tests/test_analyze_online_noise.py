from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from analyze_online_noise import (paired_predictive_summary, noise_failure_diagnostics,
                                  evaluate_future_estimate, independent_residual_examples)
from benchmarks.mrclam.evaluate import GroundTruthData, RobotGroundTruthSeries


def test_prediction_comparison_requires_identical_future_events():
    rows = [dict(method=method, robot=1, source_row=i, nll=i+offset, energy=2.)
            for method, offset in [('fixed', 1.), ('learned', 0.)]
            for i in range(3)]
    summary = paired_predictive_summary(rows)
    assert summary['learned']['paired_nll_change'] == pytest.approx(-1.)
    assert summary['learned']['predictions'] == 3
    with pytest.raises(ValueError, match='same events'):
        paired_predictive_summary(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        paired_predictive_summary(rows + [rows[0]])


def test_failure_diagnostics_groups_successive_innovations_by_identity():
    rows = [dict(method='fixed', robot=1, source_row=i, subject=6,
                 elapsed_s=i, bearing_innovation=i + 1., range_innovation=2. * (i + 1.),
                 bearing_variance=1., range_variance=1., state_bearing_variance=0.,
                 state_range_variance=0., nll=i, energy=i + 1.) for i in range(5)]
    summary = noise_failure_diagnostics(rows)
    assert len(summary) == 1
    assert summary[0]['successive_same_identity_pairs'] == 4
    assert summary[0]['bearing_innovation_lag_correlation'] == pytest.approx(1.)
    assert summary[0]['range_innovation_lag_correlation'] == pytest.approx(1.)


def test_failure_diagnostics_reports_missing_pairs_without_nan():
    row = dict(method='fixed', robot=1, subject=6, elapsed_s=0.,
               bearing_innovation=.1, range_innovation=.2,
               bearing_variance=1., range_variance=1., state_bearing_variance=0.,
               state_range_variance=0., nll=0., energy=.05)
    summary = noise_failure_diagnostics([row])[0]
    assert summary['median_pair_time_gap_s'] is None
    assert summary['bearing_innovation_lag_correlation'] is None


def test_future_state_evaluation_excludes_warmup_for_online_and_batch():
    truth = GroundTruthData(robots={1: RobotGroundTruthSeries(
        1, np.arange(4.), np.column_stack([np.arange(4.), np.zeros(4)]), np.zeros(4))}, landmarks={})
    rows = [dict(robot=1, kind='pose', index=i, timestamp_s=float(i),
                 x=float(i) if i > 1 else 100., y=0., yaw=0.) for i in range(4)]
    aggregate, _ = evaluate_future_estimate(rows, truth, {}, 1.)
    assert aggregate['pose_count'] == 2
    assert aggregate['translation_rmse_m'] == pytest.approx(0., abs=1e-10)


def test_independent_reference_examples_match_immutable_source_rows(tmp_path):
    np.savez(tmp_path / 'd4_r3_raw.npz', raw_row_index=np.array([20, 10]),
             residual=np.array([[.5, .2], [.01, .02]]), valid=np.array([True, True]))
    rows = [dict(method=method, robot=3, source_row=10, subject=6,
                 elapsed_s=65., nll=nll, bearing_innovation=.3, range_innovation=.6)
            for method, nll in [('fixed', 0.), ('profiled_conditional', 50.)]]
    examples = independent_residual_examples(rows, tmp_path, 4)
    assert examples[0]['gt_range_error'] == pytest.approx(.01)
    assert examples[0]['gt_bearing_error'] == pytest.approx(.02)
    assert examples[0]['range_innovation'] == pytest.approx(.6)


def test_independent_reference_examples_accept_selected_method_and_reject_missing(tmp_path):
    np.savez(tmp_path / 'd1_r5_raw.npz', raw_row_index=np.array([10]),
             residual=np.array([[.01, .02]]), valid=np.array([True]))
    rows = [dict(method=method, robot=5, source_row=10, subject=6,
                 elapsed_s=65., nll=nll, bearing_innovation=.3, range_innovation=.6)
            for method, nll in [('fixed', 0.), ('reml_diagonal_q', 50.)]]
    examples = independent_residual_examples(rows, tmp_path, 1, method='reml_diagonal_q')
    assert examples[0]['gt_range_error'] == pytest.approx(.01)
    with pytest.raises(ValueError, match='no predictions'):
        independent_residual_examples(rows, tmp_path, 1)
