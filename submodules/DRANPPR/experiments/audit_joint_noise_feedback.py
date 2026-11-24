"""Past-only joint odometry/observation calibration on the native failure robot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from audit_contrast_noise import calibrate_contrast
from audit_online_noise import replay, calibrate_robot
from audit_residual_evidence import sha256, write_csv
from evaluate_incremental_reference import read_rows


def calibrate(robot, method):
    return (calibrate_robot(robot, method) if method == 'profiled_conditional' else
            calibrate_contrast(robot, method)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robot', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    reference = json.loads((args.reference/'manifest.json').read_text())
    events = args.stream/'events.jsonl'
    if sha256(events) != reference['event_sha256']:
        raise ValueError('event digest mismatch')
    methods = ('fixed', 'profiled_conditional', 'reml_fixed_q', 'reml_joint_q', 'reml_diagonal_q')
    with events.open() as stream:
        result = replay((json.loads(line) for line in stream), reference['warmup_s'],
                        reference['duration_s'], methods, calibrator=calibrate,
                        robot_ids={args.robot})
    expected = {(row['method'], int(row['source_row'])): row
                for row in read_rows(args.reference/'predictions.csv')
                if int(row['robot']) == args.robot and row['method'] in methods}
    reconstructed = {(row['method'], row['source_row']): row
                     for row in result['predictions'] if row['method'] in ('fixed', 'profiled_conditional')}
    if expected.keys() != reconstructed.keys():
        raise ValueError('reference event sets differ')
    maximum = 0.
    for key, row in reconstructed.items():
        for field in ('nll', 'energy', 'bearing_innovation', 'range_innovation'):
            error = abs(row[field]-float(expected[key][field]))
            maximum = max(maximum, error)
            if not np.isclose(row[field], float(expected[key][field]), atol=1e-7, rtol=1e-8):
                raise ValueError('reference prediction reconstruction mismatch')
    for name in ('predictions', 'online_poses', 'summary'):
        write_csv(args.output/f'{name}.csv', result[name])
    (args.output/'calibration.json').write_text(json.dumps(result['calibration'], indent=2)+'\n')
    dataset = reference['dataset']
    for method, rows in result['final_estimates'].items():
        directory = args.output/method
        directory.mkdir()
        for row in rows:
            row['dataset'] = dataset
        write_csv(directory/f'dataset{dataset}_final_estimates.csv', rows)
        write_csv(directory/f'dataset{dataset}_online_estimates.csv',
                  [row for row in result['online_poses'] if row['method'] == method])
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'experiments/audit_contrast_noise.py',
             root/'experiments/audit_nonlinear_noise.py', root/'experiments/audit_online_noise.py',
             root/'experiments/audit_residual_evidence.py', root/'scripts/dranppr_variance_components.py',
             root/'scripts/dranppr_linear_evidence.py', root/'scripts/dranppr_profiled_noise.py',
             root/'scripts/dranppr_incremental.py']
    (args.output/'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in paths},
        reference_manifest_sha256=sha256(args.reference/'manifest.json'),
        dataset=dataset, robot=args.robot, start_time_s=result['start_time_s'],
        warmup_s=reference['warmup_s'], duration_s=reference['duration_s'],
        event_sha256=reference['event_sha256'], packet_count=result['packet_count'],
        maximum_reference_prediction_error=maximum,
        truth_used=False, distributed=False, relation_edits=False,
        old_factor_reweighting=False, calibration='past-only frozen 30-second parameters',
        selection='Robot 3 selected after prior failure; development diagnostic, not held-out sequence',
        timing='all calibration and dense prediction included'), indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
