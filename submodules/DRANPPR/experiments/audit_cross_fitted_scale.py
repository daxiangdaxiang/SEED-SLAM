"""Controlled audit of cross-fitted sensor-scale evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_linear_evidence import (
    cross_fitted_group_scale_evidence,
    estimate_group_measurement_scale,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(seeds, factors=40):
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        state = np.array([np.diag([.2 + i/factors, .4 + .5*i/factors])
                          for i in range(factors)])
        base = np.repeat(np.diag([1., 2.])[None], factors, axis=0)
        cases = {}
        for name, scale in [('nominal', 1.), ('group_scale_change', 8.)]:
            cases[name] = np.array([
                rng.multivariate_normal(np.zeros(2), state[i] + scale * base[i])
                for i in range(factors)])
        cases['one_wrong_relation'] = cases['nominal'].copy()
        cases['one_wrong_relation'][0] += np.array([20., -20.])
        for name, innovation in cases.items():
            inclusive = estimate_group_measurement_scale(innovation, state, base)
            crossfit = cross_fitted_group_scale_evidence(innovation, state, base)
            target_energy = crossfit.normalized_energy[0]
            rows.append(dict(
                seed=seed, case=name, inclusive_scale=inclusive.scale,
                target_excluded_scale=crossfit.excluded_scales[0],
                median_excluded_scale=float(np.median(crossfit.excluded_scales)),
                target_energy=target_energy,
                target_energy_percentile=float(np.mean(
                    crossfit.normalized_energy[1:] <= target_energy)),
                other_median_energy=float(np.median(crossfit.normalized_energy[1:]))))
    return rows


def summarize(rows):
    by_case = {name: [row for row in rows if row['case'] == name]
               for name in {row['case'] for row in rows}}
    field = lambda case, name: np.asarray([row[name] for row in by_case[case]])
    nominal = np.median(field('nominal', 'inclusive_scale'))
    return dict(
        seeds=len(by_case['nominal']), factors_per_group=40,
        nominal_median_scale=float(nominal),
        group_change_median_scale=float(np.median(field(
            'group_scale_change', 'inclusive_scale'))),
        group_change_ratio=float(np.median(field(
            'group_scale_change', 'inclusive_scale')) / nominal),
        wrong_relation_inclusive_scale_inflation=float(np.median(field(
            'one_wrong_relation', 'inclusive_scale')) / nominal),
        wrong_relation_excluded_scale_ratio=float(np.median(field(
            'one_wrong_relation', 'target_excluded_scale')) / nominal),
        wrong_relation_target_energy_percentile=float(np.median(field(
            'one_wrong_relation', 'target_energy_percentile'))))


def report(summary):
    return f'''# Cross-Fitted Sensor-Scale Audit

Across `{summary['seeds']}` controlled seeds with
`{summary['factors_per_group']}` factors per sensor group, a true group-wide
eight-fold covariance change produced a median fitted-scale ratio of
`{summary['group_change_ratio']:.2f}`. A single wrong relation inflated the
inclusive group scale by `{summary['wrong_relation_inclusive_scale_inflation']:.2f}x`,
but its leave-factor-out calibration scale remained
`{summary['wrong_relation_excluded_scale_ratio']:.2f}x` nominal. Its predictive
energy had median percentile
`{summary['wrong_relation_target_energy_percentile']:.3f}` within the group.

The experiment isolates one mechanism only. It shows that excluding a factor
from both state prediction and covariance calibration blocks two forms of
self-validation while preserving a coherent group-scale response. It is not an
end-to-end relation classifier or distributed SLAM comparison.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    rows = audit(args.seeds)
    summary = summarize(rows)
    with (args.output / 'rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report(summary))
    producer = Path(__file__).resolve()
    model = producer.parents[1] / 'scripts' / 'dranppr_linear_evidence.py'
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source_hashes={producer.name: sha256(producer),
                       str(model.relative_to(producer.parents[1])): sha256(model)},
        synthetic=True, truth_used=False, argv=sys.argv), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
