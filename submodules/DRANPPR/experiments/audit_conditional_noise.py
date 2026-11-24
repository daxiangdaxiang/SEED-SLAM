"""Held-out GT-assisted audit of a two-parameter range noise model.

This consumes raw residual audit NPZ files. Ground truth generated those
residuals, so the result diagnoses model structure; it is not an online
estimator or an end-to-end SLAM accuracy comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_conditional_noise import (
    fit_constant_measurement_variance,
    fit_range_conditional_variance,
    gaussian_mean_nll,
)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(input_dir):
    rows = []
    for path in sorted(Path(input_dir).glob('d*_r*_raw.npz')):
        fields = path.stem.split('_')
        dataset, robot = int(fields[0][1:]), int(fields[1][1:])
        data = np.load(path)
        residual = data['residual'][:, 0]
        distance = data['observations'][:, 1] - residual
        train, test = data['train'], data['test']
        if np.sum(train) < 3 or not np.any(test):
            raise ValueError(f'insufficient chronological split in {path}')
        # GT fixes the state for this structural audit, hence P_i = 0. The
        # deployable version must supply factor-excluded state uncertainty.
        constant = fit_constant_measurement_variance(residual[train])
        conditional = fit_range_conditional_variance(
            residual[train], distance[train])
        nominal_nll = gaussian_mean_nll(
            residual[test], np.full(np.sum(test), .1**2))
        constant_nll = gaussian_mean_nll(
            residual[test], np.full(np.sum(test), constant.variance))
        conditional_nll = gaussian_mean_nll(
            residual[test], conditional.measurement_variance(distance[test]))
        rows.append(dict(
            dataset=dataset, robot=robot,
            train_count=int(np.sum(train)), test_count=int(np.sum(test)),
            nominal_test_nll=nominal_nll,
            constant_variance=constant.variance,
            constant_test_nll=constant_nll,
            intercept_variance=conditional.intercept_variance,
            distance_squared_coefficient=conditional.distance_squared_coefficient,
            conditional_test_nll=conditional_nll,
            conditional_improvement_vs_constant=constant_nll-conditional_nll,
            conditional_improvement_vs_nominal=nominal_nll-conditional_nll,
            fit_success=constant.success and conditional.success))
    if not rows:
        raise ValueError('no raw audit NPZ files found')
    return rows


def summarize(rows):
    gains = np.asarray([row['conditional_improvement_vs_constant'] for row in rows])
    nominal = np.asarray([row['conditional_improvement_vs_nominal'] for row in rows])
    return dict(
        streams=len(rows), datasets=len({row['dataset'] for row in rows}),
        improved_over_constant=int(np.sum(gains > 0.)),
        median_nll_improvement_vs_constant=float(np.median(gains)),
        minimum_nll_improvement_vs_constant=float(np.min(gains)),
        improved_over_nominal=int(np.sum(nominal > 0.)),
        median_nll_improvement_vs_nominal=float(np.median(nominal)))


def report(summary):
    return f'''# Range-Conditional Noise Audit

This is a GT-assisted structural diagnostic on `{summary['streams']}` robot
streams from `{summary['datasets']}` MR.CLAM datasets, not a deployable
calibrator or a SLAM ranking. Each model is fitted on the first 30% of the
common acquisition interval and scored on the later 70% with normalized
Gaussian log likelihood.

The two-parameter model `Var(range error | d) = a + b d^2` improved held-out
NLL over a fitted per-robot constant variance on
`{summary['improved_over_constant']}/{summary['streams']}` streams. Median and
worst-stream improvements were
`{summary['median_nll_improvement_vs_constant']:.4f}` and
`{summary['minimum_nll_improvement_vs_constant']:.4f}` nats/observation. It
improved over the nominal 0.1 m standard deviation on
`{summary['improved_over_nominal']}/{summary['streams']}` streams.

This supports a small condition-aware sensor model instead of one global
robustness scale. It does not show that online residuals can be used directly:
the deployable likelihood must add factor-excluded state predictive variance,
and association alternatives must be compared before an inconsistent factor
is allowed to change the sensor model.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    rows = audit(args.input)
    summary = summarize(rows)
    write_csv(args.output / 'streams.csv', rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report(summary))
    producer = Path(__file__).resolve()
    model = producer.parents[1] / 'scripts' / 'dranppr_conditional_noise.py'
    inputs = sorted(args.input.glob('d*_r*_raw.npz'))
    manifest = dict(
        scope='GT-assisted model-structure audit; not deployable calibration',
        truth_used=True, split='pre-existing chronological 30/70 masks',
        source_hashes={str(producer.name): sha256(producer),
                       str(model.relative_to(producer.parents[1])): sha256(model)},
        input_hashes={path.name: sha256(path) for path in inputs},
        input_metadata_sha256=(sha256(args.input / 'metadata.json')
                               if (args.input / 'metadata.json').exists() else None))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
