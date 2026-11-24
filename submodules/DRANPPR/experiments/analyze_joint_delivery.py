"""Posthoc state-age diagnostic under the measured serial service queue."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest
from experiments.audit_native_checkpoint import sha256
from experiments.evaluate_incremental_reference import read_rows


def delivery_metrics(rows, updates, truth, alignment):
    indexed = {(int(r['robot']), float(r['timestamp_s'])):r for r in updates}
    if len(indexed) != len(updates):
        raise ValueError('duplicate frame in delivery ledger')
    c, s = math.cos(alignment['alignment_yaw']), math.sin(alignment['alignment_yaw'])
    rotation = np.array([[c,-s],[s,c]])
    translation = np.array([alignment['alignment_x'],alignment['alignment_y']])
    errors, stamped, ages = [], [], []
    unsupported = 0
    for row in rows:
        robot, t = int(row['robot']), float(row['timestamp_s'])
        age = float(indexed[robot,t]['serial_backlog_s'])
        if not math.isfinite(age) or age < 0:
            raise ValueError('invalid service age')
        series = truth.robots[robot]
        if not series.supports(t) or not series.supports(t+age):
            unsupported += 1
            continue
        estimate = rotation @ [float(row['x']),float(row['y'])]+translation
        stamped.append(float(np.linalg.norm(estimate-series.interpolate(t)[0])))
        errors.append(float(np.linalg.norm(estimate-series.interpolate(t+age)[0])))
        ages.append(age)
    if not errors:
        raise ValueError('no supported delivery timestamps')
    return dict(poses=len(errors), unsupported_poses=unsupported,
        measurement_time_rmse_m=float(np.sqrt(np.mean(np.square(stamped)))),
        delivery_time_stale_pose_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
        state_age_mean_s=float(np.mean(ages)), state_age_p99_s=float(np.quantile(ages,.99)),
        state_age_max_s=max(ages),
        assumption='Same measurement-tail alignment for both errors; no motion extrapolation. Virtual single-processor completion time, zero network latency; not measured multi-robot wall time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.results/'evaluation.json').read_text())
    root = Path(__file__).resolve().parents[1]
    output = dict(script_sha256=sha256(__file__), evaluation_sha256=sha256(args.results/'evaluation.json'),results=[])
    for run in evaluation['results']:
        if not run['mode'].startswith('joint') or 'causal_connected_tail_global_alignment' not in run:
            continue
        dataset=run['dataset']
        folder=args.results/f"d{dataset}_{int(run['duration'])}_{run['mode']}"
        manifest=load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audit=audit_dataset(manifest,args.raw)
        truth=load_ground_truth(manifest,audit.dataset_directory)
        rows=read_rows(folder/'causal_estimates.csv')
        first=min(float(r['timestamp_s']) for r in rows)
        tail=run.get('matched_tail_start_elapsed_s',run['all_connected_elapsed_s'])
        rows=[r for r in rows if float(r['timestamp_s'])-first >= tail]
        result=delivery_metrics(rows,read_rows(folder/'updates.csv'),truth,run['causal_connected_tail_global_alignment'])
        output['results'].append(dict(dataset=dataset, mode=run['mode'], duration=run['duration'], **result))
    with args.output.open('x') as target:
        json.dump(output,target,indent=2,allow_nan=False)
        target.write('\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
