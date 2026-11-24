"""Freeze an online-noise experiment's factors and audit nonlinear stationarity."""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sys
import time

import gtsam
import numpy as np

from audit_online_noise import IncrementalPoint2Robot, variance_model, METHODS
from audit_residual_evidence import final_estimate_rows, linearize, sha256, write_csv
from evaluate_incremental_reference import read_rows


def replay_fixed_calibration(packets, calibration, warmup, duration, methods=METHODS):
    robots, times, first = {method: {} for method in methods}, {}, None
    fits = {(row['method'], row['robot']): row for row in calibration}
    calibrated = set()
    for _, group in groupby(packets, key=lambda packet: (packet['robot'], packet['timestamp_s'])):
        frame = tuple(group)
        robot_id, timestamp = frame[0]['robot'], frame[0]['timestamp_s']
        if first is None:
            first = timestamp
        elapsed = timestamp - first
        if elapsed > duration:
            break
        for packet in frame:
            if packet['kind'] == 'pose':
                times[(robot_id, packet['pose_index'])] = timestamp
        for method in methods:
            if robot_id not in robots[method]:
                robots[method][robot_id] = IncrementalPoint2Robot(robot_id)
            robot = robots[method][robot_id]
            identity = (method, robot_id)
            if elapsed > warmup and method != 'fixed' and identity not in calibrated:
                fit = fits[identity]
                robot.observation_variance_model = variance_model(fit)
                if 'odometry_variance_scales' in fit:
                    scales = np.asarray(fit['odometry_variance_scales'], dtype=float)
                    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0.):
                        raise ValueError('odometry variance scales must be positive')
                    robot.odometry_sigma *= np.sqrt(scales)
                calibrated.add(identity)
            robot.push_batch(frame)
    return robots, times


def estimate_rows(values, robot, times, dataset):
    rows = []
    for key in values.keys():
        symbol = gtsam.Symbol(key)
        index = symbol.index()
        if symbol.chr() == ord('x'):
            pose = values.atPose2(key)
            rows.append(dict(dataset=dataset, robot=robot, kind='pose', index=index,
                             timestamp_s=times[(robot, index)], x=pose.x(), y=pose.y(), yaw=pose.theta()))
        else:
            point = values.atPoint2(key)
            rows.append(dict(dataset=dataset, robot=robot, kind='landmark', index=index,
                             timestamp_s='', x=point[0], y=point[1], yaw=''))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-state-initialization', action='store_true',
                        help='diagnostic: initialize the same frozen graph from the fixed-noise arm')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    events = args.stream / 'events.jsonl'
    if sha256(events) != manifest['event_sha256']:
        raise ValueError('event digest mismatch')
    calibration = json.loads((args.run / 'calibration.json').read_text())
    with events.open() as stream:
        robots, times = replay_fixed_calibration(
            (json.loads(line) for line in stream), calibration,
            manifest['warmup_s'], manifest['duration_s'])
    dataset = manifest['dataset']
    results = []
    for method, local in robots.items():
        expected = {(int(row['robot']), row['kind'], int(row['index'])): row for row in
                    read_rows(args.run / method / f'dataset{dataset}_final_estimates.csv')}
        reconstructed = final_estimate_rows(local, times, dataset)
        differences = [np.linalg.norm(np.array([row['x'], row['y']]) - np.array([
            float(expected[(row['robot'], row['kind'], row['index'])][key]) for key in ('x', 'y')]))
                       for row in reconstructed]
        reconstruction_error = float(max(differences))
        if reconstruction_error > 1e-8:
            raise ValueError(f'reconstruction mismatch: {method}: {reconstruction_error}')
        batch_rows = []
        for robot_id, robot in local.items():
            values = robot.graph.estimate()
            graph = robot.graph.factor_graph()
            physical = robot.physical_graph()
            a, r, _ = linearize(physical, values)
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(100)
            params.setRelativeErrorTol(1e-10)
            params.setAbsoluteErrorTol(1e-10)
            initial = (robots['fixed'][robot_id].graph.estimate()
                       if args.fixed_state_initialization else values)
            start = time.perf_counter()
            solver = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            batch = solver.optimize()
            wall = time.perf_counter() - start
            a1, r1, _ = linearize(physical, batch)
            results.append(dict(method=method, robot=robot_id,
                                reconstruction_error=reconstruction_error,
                                online_cost=float(physical.error(values)),
                                solve_initial_cost=float(physical.error(initial)),
                                batch_cost=float(physical.error(batch)),
                                online_gradient_norm=float(np.linalg.norm(a.T @ r)),
                                batch_gradient_norm=float(np.linalg.norm(a1.T @ r1)),
                                batch_iterations=solver.iterations(), batch_s=wall))
            batch_rows.extend(estimate_rows(batch, robot_id, times, dataset))
        target = args.output / method
        target.mkdir()
        write_csv(target / f'dataset{dataset}_final_estimates.csv', batch_rows)
    write_csv(args.output / 'stationarity.csv', results)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_sha256=sha256(Path(__file__)),
        producer_manifest_sha256=sha256(args.run / 'manifest.json'),
        fixed_state_initialization=args.fixed_state_initialization,
        old_factor_noise_changed=False, posthoc_full_history=True, truth_used=False), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
