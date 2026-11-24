"""Q-only/R-only attribution and fixed-procedure multi-robot noise validation.

Independent local references only; sparse covariance still refactorizes each
current graph and must not be described as an incremental distributed backend.
"""
import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
import scipy

from audit_contrast_noise import calibrate_contrast
from audit_online_noise import replay, calibrate_robot, predicted_observations
from audit_residual_evidence import sha256, write_csv
from dranppr_sparse_prediction import sparse_selected_covariance
from dranppr_sparse_linearization import sparse_linearize
from evaluate_incremental_reference import read_rows

METHODS = ('fixed', 'fitted_residual', 'q_only', 'r_only', 'reml_diagonal_q')


def ablate_calibration(fit, method):
    if method not in ('q_only', 'r_only', 'reml_diagonal_q'):
        raise ValueError('unknown factorial arm')
    result = deepcopy(fit)
    result.update(method=method, unablated_fit=deepcopy(fit))
    if method == 'q_only':
        result.update(bearing_variance=.05**2, range_intercept_variance=.1**2,
                      range_distance_squared_coefficient=0.)
    elif method == 'r_only':
        result['odometry_variance_scales'] = [1., 1., 1.]
    return result


def calibrate(robot, method):
    result = (calibrate_robot(robot, method) if method == 'fitted_residual' else
              ablate_calibration(calibrate_contrast(robot, 'reml_diagonal_q')[0], method))
    print(json.dumps(dict(robot=robot.robot, method=method, success=result['success'],
                          bearing_variance=result['bearing_variance'],
                          odometry_variance_scales=result.get('odometry_variance_scales', [1., 1., 1.]))),
          flush=True)
    return result


def compare_reference(rows, reference_rows, reference_name):
    key = lambda row: (row['method'], int(row['robot']), int(row['source_row']))
    current = {key(row): row for row in rows}
    reference = {key(row): row for row in reference_rows}
    current_groups = {k[:2] for k in current}
    common = current_groups & {k[:2] for k in reference}
    if not common:
        raise ValueError('no shared reference events')
    current = {k: r for k, r in current.items() if k[:2] in common}
    reference = {k: r for k, r in reference.items() if k[:2] in common}
    if current.keys() != reference.keys():
        raise ValueError('reference event sets differ')
    result = []
    for field in ('nll', 'energy', 'bearing_innovation', 'range_innovation',
                  'state_bearing_variance', 'state_range_variance', 'state_cross_covariance'):
        actual = np.array([float(row[field]) for row in current.values()])
        expected = np.array([float(reference[k][field]) for k in current])
        result.append(dict(reference=reference_name, field=field, events=len(actual),
                           max_absolute_error=float(np.max(np.abs(actual-expected))),
                           relative_l2_error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-15)),
                           passed=bool(np.allclose(actual, expected, rtol=1e-7, atol=1e-9))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--robots', type=int, nargs='+')
    parser.add_argument('--warmup', type=float, default=30.)
    parser.add_argument('--duration', type=float, default=120.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    metadata = json.loads((args.stream/'metadata.json').read_text())
    events = args.stream/'events.jsonl'
    if sha256(events) != metadata['event_sha256']:
        raise ValueError('event digest mismatch')
    predictor = partial(predicted_observations, covariance_solver=sparse_selected_covariance,
                        linearizer=sparse_linearize)
    with events.open() as stream:
        result = replay((json.loads(line) for line in stream), args.warmup, args.duration,
                        METHODS, calibrator=calibrate, predictor=predictor,
                        robot_ids=set(args.robots) if args.robots else None)
    for name in ('predictions', 'online_poses', 'summary'):
        write_csv(args.output/f'{name}.csv', result[name])
    (args.output/'calibration.json').write_text(json.dumps(result['calibration'], indent=2)+'\n')
    dataset = metadata['dataset']
    for method, rows in result['final_estimates'].items():
        directory = args.output/method
        directory.mkdir()
        for row in rows:
            row['dataset'] = dataset
        write_csv(directory/f'dataset{dataset}_final_estimates.csv', rows)
        write_csv(directory/f'dataset{dataset}_online_estimates.csv',
                  [row for row in result['online_poses'] if row['method'] == method])
    comparisons = []
    for directory in args.reference:
        reference = json.loads((directory/'manifest.json').read_text())
        if reference['event_sha256'] != metadata['event_sha256']:
            raise ValueError('reference event digest differs')
        comparisons.extend(compare_reference(result['predictions'], read_rows(directory/'predictions.csv'), str(directory)))
    if comparisons:
        write_csv(args.output/'reference_parity.csv', comparisons)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'experiments/audit_contrast_noise.py', root/'experiments/audit_nonlinear_noise.py',
             root/'experiments/audit_online_noise.py', root/'experiments/audit_residual_evidence.py',
             root/'scripts/dranppr_variance_components.py', root/'scripts/dranppr_linear_evidence.py',
             root/'scripts/dranppr_profiled_noise.py', root/'scripts/dranppr_conditional_noise.py',
             root/'scripts/dranppr_incremental.py', root/'scripts/dranppr_sparse_prediction.py',
             root/'scripts/dranppr_sparse_linearization.py']
    (args.output/'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in paths},
        reference_manifest_hashes={str(p): sha256(p/'manifest.json') for p in args.reference},
        dataset=dataset, robots=args.robots, start_time_s=result['start_time_s'],
        warmup_s=args.warmup, duration_s=args.duration, event_sha256=metadata['event_sha256'],
        packet_count=result['packet_count'], reference_parity_passed=all(r['passed'] for r in comparisons),
        reference_parity_fields=len(comparisons), numpy_version=np.__version__, scipy_version=scipy.__version__,
        truth_used=False, distributed=False, relation_edits=False, old_factor_reweighting=False,
        covariance='native sparse Jacobian; fresh sparse augmented-system LU; complete queried cross covariance; no normal equations',
        calibration='identical past-only fit for Q-only/R-only/joint, then frozen; each arm pays its own fit cost',
        selection='fixed procedure after the selected D4 Robot 3 development case'), indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2), flush=True)
    if not all(row['passed'] for row in comparisons):
        raise RuntimeError('sparse predictions do not pass dense-reference parity')


if __name__ == '__main__':
    main()
