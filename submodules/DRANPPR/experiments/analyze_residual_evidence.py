"""Summarize a completed residual-evidence audit without changing raw traces."""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _mean(rows, field):
    if not rows:
        raise ValueError(f'no rows for {field}')
    return float(np.mean([float(row[field]) for row in rows]))


def _ratios(groups, variant, field):
    result = []
    for variants in groups.values():
        if variant not in variants or 'batch_refined' not in variants:
            continue
        denominator = float(variants['batch_refined'][field])
        if denominator > 0:
            result.append(float(variants[variant][field]) / denominator)
    if not result:
        raise ValueError(f'no paired {variant} rows')
    return np.asarray(result)


def _ratio_summary(values):
    return dict(count=len(values), median_ratio=float(np.median(values)),
                p05_ratio=float(np.quantile(values, .05)),
                p95_ratio=float(np.quantile(values, .95)),
                maximum_ratio=float(np.max(values)))


def analyze(path):
    path = Path(path)
    synthetic = _rows(path / 'synthetic.csv')
    unchanged = [row for row in synthetic if row['case'] == 'unchanged_data']
    clean = [row for row in unchanged if float(row['initial_error_m']) == 0. and
             int(row['iterations']) == 0]
    inaccurate = [row for row in unchanged if float(row['initial_error_m']) == 1. and
                   int(row['iterations']) == 0]
    corrupt = [row for row in synthetic if row['case'] == 'one_corrupt_observation']
    equivalent = [row for row in synthetic if
                  row['case'] == 'unidentifiable_state_equivalent_bias']
    seeds = sorted({row['seed'] for row in unchanged})
    clean_by_seed = {seed: [row for row in clean if row['seed'] == seed]
                     for seed in seeds}
    inaccurate_by_seed = {seed: [row for row in inaccurate if row['seed'] == seed]
                           for seed in seeds}
    corrupt_by_seed = {seed: [row for row in corrupt if row['seed'] == seed]
                       for seed in seeds}
    equivalent_by_seed = {seed: [row for row in equivalent if row['seed'] == seed]
                          for seed in seeds}
    naive_inflation = [_mean(inaccurate_by_seed[seed], 'fitted_ml_scale2') /
                       _mean(clean_by_seed[seed], 'fitted_ml_scale2') for seed in seeds]
    contrast_spread = []
    corrupt_ratio, equivalent_ratio = [], []
    for seed in seeds:
        same_seed = [float(row['scale_squared']) for row in unchanged
                     if row['seed'] == seed]
        contrast_spread.append(max(same_seed) - min(same_seed))
        baseline = _mean(clean_by_seed[seed], 'scale_squared')
        corrupt_ratio.append(_mean(corrupt_by_seed[seed], 'scale_squared') / baseline)
        equivalent_ratio.append(_mean(equivalent_by_seed[seed], 'scale_squared') / baseline)
    synthetic_summary = dict(
        seeds=len(seeds),
        naive_initialization_inflation=float(np.mean(naive_inflation)),
        contrast_initialization_spread=float(np.max(contrast_spread)),
        localized_corruption_contrast_ratio=float(np.mean(corrupt_ratio)),
        state_equivalent_bias_contrast_ratio=float(np.mean(equivalent_ratio)))

    native_rows = []
    for filename in glob.glob(str(path / 'dataset*_evidence.csv')):
        native_rows.extend(_rows(Path(filename)))
    groups = {}
    for row in native_rows:
        key = (int(row['dataset']), int(row['robot']), float(row['elapsed_s']))
        groups.setdefault(key, {})[row['variant']] = row
    small_naive = _ratios(groups, 'state_perturbed_01', 'fitted_ml_scale2')
    small_contrast = _ratios(groups, 'state_perturbed_01', 'scale_squared')
    large_naive = _ratios(groups, 'state_perturbed_10', 'fitted_ml_scale2')
    large_contrast = _ratios(groups, 'state_perturbed_10', 'scale_squared')
    incremental_contrast = _ratios(groups, 'incremental', 'scale_squared')
    native_summary = dict(
        paired_checkpoints=len(groups),
        incremental=dict(**_ratio_summary(incremental_contrast),
                         contrast_max_relative_difference=float(
                             np.max(np.abs(incremental_contrast - 1.)))),
        small_perturbation=dict(
            naive_median_ratio=float(np.median(small_naive)),
            contrast_median_ratio=float(np.median(small_contrast)),
            contrast_p95_ratio=float(np.quantile(small_contrast, .95)),
            contrast_maximum_ratio=float(np.max(small_contrast))),
        large_perturbation=dict(
            naive_median_ratio=float(np.median(large_naive)),
            contrast_median_ratio=float(np.median(large_contrast)),
            contrast_p95_ratio=float(np.quantile(large_contrast, .95))))

    predictive_rows = [row for row in _rows(path / 'predictive.csv')
                       if row['identifiable'].lower() == 'true']
    prediction = {(row['case'], row['seed'], row['block']):
                  float(row['energy_per_dimension']) for row in predictive_rows}
    nominal = {(row['seed'], row['block']): float(row['energy_per_dimension'])
               for row in predictive_rows if row['case'] == 'nominal'}
    equivalent_difference = [abs(float(row['energy_per_dimension']) -
                                 nominal[(row['seed'], row['block'])])
                             for row in predictive_rows
                             if row['case'] == 'state_equivalent_bias']
    prediction_seeds = sorted({row['seed'] for row in predictive_rows})
    degraded_ratio, unaffected_ratio = [], []
    wrong_ratio, wrong_percentile = [], []
    for seed in prediction_seeds:
        def selected(case, group=None, target=None):
            values = []
            for row in predictive_rows:
                if row['seed'] != seed or row['case'] != case:
                    continue
                if group is not None and row['group'] != group:
                    continue
                if target is not None and (row['target'].lower() == 'true') != target:
                    continue
                values.append(float(row['energy_per_dimension']))
            if not values:
                raise ValueError(f'missing predictive rows for {case}, {group}, {target}')
            return np.asarray(values)
        degraded_ratio.append(float(np.median(selected(
            'group_sensor_degradation', 'object_sensor_0')) /
            np.median(selected('nominal', 'object_sensor_0'))))
        unaffected_ratio.append(float(np.median(selected(
            'group_sensor_degradation', 'object_sensor_1')) /
            np.median(selected('nominal', 'object_sensor_1'))))
        target_value = float(selected('one_wrong_relationship', target=True)[0])
        other = selected('one_wrong_relationship', target=False)
        wrong_ratio.append(target_value / float(np.median(other)))
        wrong_percentile.append(float(np.mean(other <= target_value)))
    predictive_summary = dict(
        identifiable_blocks=len(nominal),
        state_equivalent_max_absolute_difference=float(max(equivalent_difference)),
        degraded_group_median_ratio=float(np.median(degraded_ratio)),
        degraded_unaffected_median_ratio=float(np.median(unaffected_ratio)),
        wrong_relationship_target_to_other_median=float(np.median(wrong_ratio)),
        wrong_relationship_target_percentile=float(np.median(wrong_percentile)))

    native_predictive_rows = []
    per_dataset_predictive = {}
    for filename in glob.glob(str(path / 'dataset*_predictive.csv')):
        rows = _rows(Path(filename))
        if not rows:
            continue
        last_time = max(float(row['elapsed_s']) for row in rows)
        latest = [row for row in rows if float(row['elapsed_s']) == last_time]
        native_predictive_rows.extend(latest)
        per_dataset_predictive[str(latest[0]['dataset'])] = dict(
            elapsed_s=last_time, factors=len(latest),
            identifiable=sum(row['identifiable'].lower() == 'true'
                             for row in latest))
    native_predictive_summary = {}
    for kind in ('observation', 'odometry'):
        rows = [row for row in native_predictive_rows
                if row['factor_kind'] == kind]
        identifiable = [row for row in rows
                        if row['identifiable'].lower() == 'true']
        energy = np.asarray([float(row['energy_per_dimension'])
                             for row in identifiable])
        if not rows or not len(energy):
            raise ValueError(f'no identifiable native predictive rows for {kind}')
        native_predictive_summary[kind] = dict(
            count=len(rows), identifiable_count=len(identifiable),
            identifiable_fraction=len(identifiable) / len(rows),
            median_energy_per_dimension=float(np.median(energy)),
            mean_energy_per_dimension=float(np.mean(energy)),
            p95_energy_per_dimension=float(np.quantile(energy, .95)))
    native_predictive_summary['per_dataset'] = per_dataset_predictive

    manifest = json.loads((path / 'manifest.json').read_text())
    timings = manifest['summaries']
    timing_summary = dict(
        datasets=len(timings), total_updates=sum(int(row['updates']) for row in timings),
        maximum_p50_ms=max(float(row['p50_ms']) for row in timings),
        maximum_p95_ms=max(float(row['p95_ms']) for row in timings),
        maximum_p99_ms=max(float(row['p99_ms']) for row in timings),
        maximum_single_update_ms=max(float(row['max_ms']) for row in timings),
        maximum_serial_backlog_s=max(float(row['max_serial_backlog_s']) for row in timings))
    return dict(synthetic=synthetic_summary, predictive=predictive_summary,
                native_predictive=native_predictive_summary,
                native=native_summary,
                timing=timing_summary, per_dataset_timing=timings)


def report(summary):
    s, p = summary['synthetic'], summary['predictive']
    npred = summary['native_predictive']
    n, t = summary['native'], summary['timing']
    return f'''# Residual Evidence and Incremental Reference Audit

## What Was Tested

The controlled linear experiment changes initialization and iterative solve
budget while holding measurements, associations and reported covariance fixed.
The native experiment replays the first 60 seconds of five original MR.CLAM
streams through persistent, independent local iSAM2 graphs. It does not inject
noise, read evaluation truth, perform inter-robot registration or run DRANPPR.

## Controlled Result

- A one-meter initial state error inflated the naive residual scale by
  `{s['naive_initialization_inflation']:.1f}x` on average.
- The state-orthogonal contrast scale changed by at most
  `{s['contrast_initialization_spread']:.3g}` across all initializations and
  linear CG budgets.
- One localized corrupted observation raised contrast scale by
  `{s['localized_corruption_contrast_ratio']:.2f}x`.
- A deliberately state-equivalent bias changed contrast scale by only
  `{s['state_equivalent_bias_contrast_ratio']:.3f}x`. It is unidentifiable from
  these measurements and must not be advertised as detectable.

Factor-excluded prediction gives the same score under that state-equivalent
change to within `{p['state_equivalent_max_absolute_difference']:.3g}`. A
group-wide sensor degradation raised the affected sensor group's median score by
`{p['degraded_group_median_ratio']:.2f}x`, while the other robot's object sensor
changed by `{p['degraded_unaffected_median_ratio']:.2f}x`. A single wrong
relationship had a target-to-other median score ratio of
`{p['wrong_relationship_target_to_other_median']:.1f}x` and a median rank
percentile of `{p['wrong_relationship_target_percentile']:.3f}`. These are
population/localization patterns, not a proof that every individual cause is
identifiable.

## Native Nonlinear Result

Across `{n['paired_checkpoints']}` robot-time checkpoints, persistent incremental
and batch-refined contrast scales differed by at most
`{n['incremental']['contrast_max_relative_difference']:.3g}` relatively. A
small state perturbation inflated raw residual scale by a median
`{n['small_perturbation']['naive_median_ratio']:.1f}x`, while contrast scale
changed by a median `{n['small_perturbation']['contrast_median_ratio']:.3f}x`
and a 95th percentile `{n['small_perturbation']['contrast_p95_ratio']:.3f}x`.

A large perturbation is a clear boundary: even contrast scale changed by a
median `{n['large_perturbation']['contrast_median_ratio']:.1f}x`. The fixed local
linearization cannot remove large nonlinear state error. It is evidence for
using a trust-region response or reporting ambiguity, not for adapting sensor
noise from that residual.

At each dataset's final checkpoint, factor deletion left
`{npred['observation']['identifiable_fraction']:.1%}` of object observations and
`{npred['odometry']['identifiable_fraction']:.1%}` of odometry factors
independently predictable in the local linear model. The remaining factors are
reported as unidentifiable, not assigned a finite goodness score. Identifiable
object and odometry median energies per dimension were
`{npred['observation']['median_energy_per_dimension']:.3f}` and
`{npred['odometry']['median_energy_per_dimension']:.3f}`. These values describe
the full sensing/model/reference stack and correlated graph constraints; they
are not direct sensor variance estimates.

## Incremental Timing

The local reference processed `{t['total_updates']}` causal events. The worst
dataset p99 was `{t['maximum_p99_ms']:.3f} ms`, the slowest single event was
`{t['maximum_single_update_ms']:.3f} ms`, and maximum simulated serial backlog
was `{t['maximum_serial_backlog_s']:.4f} s`. Dense SVD audits and batch controls
were excluded from these update timings.

## Valid Claim Boundary

The projection separates residual energy explainable by a fixed local state
linearization from state-orthogonal inconsistency. It does not by itself decide
whether the latter is sensor noise, model bias or a wrong relationship. That
requires factor-excluded predictive evidence and independent identity evidence.
The timing result is an independent local baseline, not distributed DRANPPR
performance and not an accuracy comparison.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.results)
    (args.results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.results / 'REPORT.md').write_text(report(summary))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
