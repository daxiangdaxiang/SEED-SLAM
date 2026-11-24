"""Post-selected failure audit: refine past states before a held-out sensor frame.

No claim of prospective validation follows from selecting a known failing frame.
The intervention keeps every past factor and its original frozen covariance.
"""
import argparse
from copy import copy
from itertools import groupby
import json
from pathlib import Path
import sys
import time

import gtsam
import numpy as np

from audit_online_lower import replay_fixed_calibration
from audit_online_noise import predicted_observations
from audit_residual_evidence import linearize, sha256, write_csv
from evaluate_incremental_reference import read_rows


def prefix_before_source(packets, robot, source_row):
    prefix, first = [], None
    for _, group in groupby(packets, key=lambda p: (p['robot'], p['timestamp_s'])):
        frame = tuple(group)
        if first is None:
            first = frame[0]['timestamp_s']
        if any(p['robot'] == robot and p['kind'] == 'observation' and
               p['source_row'] == source_row for p in frame):
            return prefix, frame, frame[0]['timestamp_s'] - first
        prefix.extend(frame)
    raise ValueError('target source row not found')


def prediction_at_values(robot, frame, values):
    view = copy(robot)
    view.graph = copy(robot.graph)
    view.graph.estimate = lambda: gtsam.Values(values)
    return predicted_observations(view, frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--source-row', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    events = args.stream / 'events.jsonl'
    if sha256(events) != manifest['event_sha256']:
        raise ValueError('event digest mismatch')
    with events.open() as stream:
        prefix, frame, elapsed = prefix_before_source(
            (json.loads(line) for line in stream), args.robot, args.source_row)
    calibration = json.loads((args.run / 'calibration.json').read_text())
    methods = ('fixed', *dict.fromkeys(row['method'] for row in calibration))
    robots, _ = replay_fixed_calibration(
        prefix, calibration, manifest['warmup_s'], manifest['duration_s'], methods)
    expected = {(row['method'], int(row['source_row'])): row for row in
                read_rows(args.run / 'predictions.csv') if int(row['robot']) == args.robot}
    results = []
    for method, local in robots.items():
        robot = local[args.robot]
        online = robot.graph.estimate()
        physical = robot.physical_graph()
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-10)
        params.setAbsoluteErrorTol(1e-10)
        start = time.perf_counter()
        solver = gtsam.LevenbergMarquardtOptimizer(robot.graph.factor_graph(), online, params)
        batch = solver.optimize()
        solve_s = time.perf_counter() - start
        for state, values in [('online', online), ('past_refined', batch)]:
            a, residual, _ = linearize(physical, values)
            start = time.perf_counter()
            predictions = prediction_at_values(robot, frame, values)
            prediction_s = time.perf_counter() - start
            for row in predictions:
                covariance = row.pop('state_covariance')
                if state == 'online':
                    for key in ('nll', 'energy', 'range_innovation', 'bearing_innovation'):
                        if not np.isclose(row[key], float(expected[(method, row['source_row'])][key]),
                                          atol=1e-7, rtol=1e-8):
                            raise ValueError('checkpoint does not reconstruct original prediction')
                results.append(dict(
                    method=method, state=state, robot=args.robot, elapsed_s=elapsed,
                    past_cost=float(physical.error(values)),
                    gradient_norm=float(np.linalg.norm(a.T @ residual)),
                    batch_iterations=solver.iterations(), batch_s=solve_s,
                    prediction_s=prediction_s, state_bearing_variance=float(covariance[0, 0]),
                    state_range_variance=float(covariance[1, 1]), **row))
    write_csv(args.output / 'predictions.csv', results)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'experiments/audit_online_lower.py',
               root / 'experiments/audit_online_noise.py', root / 'scripts/dranppr_incremental.py']
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in sources},
        producer_manifest_sha256=sha256(args.run / 'manifest.json'),
        selection='known failure chosen post hoc; diagnostic only', truth_used=False,
        target_frame_assimilated=False, old_factor_noise_changed=False,
        interventions='one past-only batch solve; same frozen graph and initialization'), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
