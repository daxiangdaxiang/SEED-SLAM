"""Chronological local-reference noise adaptation; not distributed DRANPPR.

Calibrate on a past-only warmup, freeze parameters, and score future observations
before assimilation. All prediction and calibration work is explicitly timed.
Ground truth is never imported by this producer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys
import time

import gtsam
import numpy as np
from scipy.linalg import qr, solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_incremental import IncrementalPoint2Robot
from dranppr_profiled_noise import DeletedLinearLikelihood, fit_polar_noise
from dranppr_conditional_noise import fit_constant_measurement_variance, fit_range_conditional_variance
from audit_residual_evidence import linearize, final_estimate_rows, sha256, write_csv

METHODS = ('fixed', 'fitted_residual', 'profiled_diagonal', 'profiled_conditional')


def variance_model(parameters):
    def variance(distance):
        return np.array([
            parameters['bearing_variance'],
            parameters['range_intercept_variance'] +
            parameters['range_distance_squared_coefficient'] * distance**2])
    return variance


def calibrate_robot(robot, method):
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-10)
    params.setAbsoluteErrorTol(1e-10)
    values = gtsam.LevenbergMarquardtOptimizer(
        robot.graph.factor_graph(), robot.graph.estimate(), params).optimize()
    a, residual, _ = linearize(robot.physical_graph(), values)
    items = robot.physical_factor_items()
    sizes = [factor.dim() for _, factor in items]
    obs = [i for i, (key, _) in enumerate(items) if key[0] == 'observation']
    predictor = DeletedLinearLikelihood(a, -residual, sizes, obs)
    predictions = predictor.predictions(np.ones(len(residual)))
    # Fix the distance covariate from a prediction that excluded this observation.
    distances = np.array([items[i][1].measured().range() - .1 * innovation[1]
                          for i, innovation, _ in predictions])
    selected = [i for i, _, _ in predictions]
    if method in ('profiled_diagonal', 'profiled_conditional'):
        result = fit_polar_noise(a, -residual, sizes, selected, distances,
                                conditional=method == 'profiled_conditional')
    elif method == 'fitted_residual':
        rows = np.array([predictor.slices[i].start for i in selected])
        bearing = fit_constant_measurement_variance(.05 * residual[rows])
        distance = fit_range_conditional_variance(.1 * residual[rows + 1], distances)
        result = dict(bearing_variance=bearing.variance,
                      range_intercept_variance=distance.intercept_variance,
                      range_distance_squared_coefficient=distance.distance_squared_coefficient,
                      success=bearing.success and distance.success,
                      at_numerical_boundary=False)
    else:
        raise ValueError('unsupported calibration method')
    result.update(robot=robot.robot, method=method,
                  training_observations=len(obs), scored_observations=len(selected),
                  unobservable_observations=len(obs) - len(selected),
                  minimum_training_range=float(np.min(distances)),
                  maximum_training_range=float(np.max(distances)),
                  fixed_linearization=True)
    return result


def predicted_observations(robot, frame, covariance_solver=None, linearizer=linearize):
    known = [packet for packet in frame if packet['kind'] == 'observation' and
             gtsam.symbol('l', packet['landmark_subject']) in robot.graph.keys]
    if robot.pose_index < 0 or not known:
        return []
    values = robot.graph.estimate()
    parent = gtsam.symbol('x', robot.pose_index)
    old_pose = values.atPose2(parent)
    pose = old_pose
    f, g, q = np.eye(3), np.eye(3), np.zeros((3, 3))
    if frame[0]['kind'] == 'pose':
        packet = frame[0]
        relative = gtsam.Pose2(*packet['delta'])
        f = relative.inverse().AdjointMap()
        pose = old_pose.compose(relative)
        q = np.diag(robot.odometry_sigma**2 * packet['duration_s'])
    # Root is exactly fixed; no finite gauge prior or normal equations are used.
    a, _, graph_keys = linearizer(robot.physical_graph(), values)
    columns, offset = {}, 0
    for key in graph_keys:
        dimension = 3 if gtsam.Symbol(key).chr() == ord('x') else 2
        columns[key] = list(range(offset, offset + dimension))
        offset += dimension
    root = gtsam.symbol('x', 0)
    keep = np.array([i for key in graph_keys if key != root for i in columns[key]])
    mapping = {old: new for new, old in enumerate(keep)}
    requested = [parent, *dict.fromkeys(gtsam.symbol('l', packet['landmark_subject']) for packet in known)]
    rhs = np.zeros((len(keep), sum(len(columns[key]) for key in requested)))
    requested_slices, start = {}, 0
    for key in requested:
        dimension = len(columns[key])
        requested_slices[key] = slice(start, start + dimension)
        if key != root:
            rhs[[mapping[col] for col in columns[key]], np.arange(start, start + dimension)] = 1.
        start += dimension
    if covariance_solver is None:
        _, triangular = qr(a[:, keep], mode='economic', check_finite=False)
        selected_inverse = solve_triangular(triangular.T, rhs, lower=True, check_finite=False)
        joint_covariance = selected_inverse.T @ selected_inverse
    else:
        joint_covariance = covariance_solver(a[:, keep], rhs)
    prediction_values = gtsam.Values(values)
    prediction_values.update(parent, pose)
    output = []
    for packet in known:
        landmark = gtsam.symbol('l', packet['landmark_subject'])
        keys = [parent, landmark]
        covariance = np.block([[joint_covariance[requested_slices[a], requested_slices[b]]
                                for b in keys] for a in keys])
        propagation = np.eye(5)
        propagation[:3, :3] = f
        covariance = propagation @ covariance @ propagation.T
        covariance[:3, :3] += g @ q @ g.T
        point = values.atPoint2(landmark)
        distance = float(pose.range(point))
        factor = gtsam.BearingRangeFactor2D(
            parent, landmark, gtsam.Rot2(packet['bearing_rad']), packet['range_m'],
            gtsam.noiseModel.Unit.Create(2))
        ordering = gtsam.Ordering()
        for key in keys:
            ordering.push_back(key)
        linear = gtsam.GaussianFactorGraph()
        linear.push_back(factor.linearize(prediction_values))
        jacobian, _ = linear.jacobian(ordering)
        innovation = -factor.unwhitenedError(prediction_values)
        state_covariance = jacobian @ covariance @ jacobian.T
        variance = (robot.observation_noise.sigmas()**2
                    if robot.observation_variance_model is None else
                    robot.observation_variance_model(distance))
        total = state_covariance + np.diag(variance)
        chol = np.linalg.cholesky(total)
        whitened = np.linalg.solve(chol, innovation)
        energy = float(whitened @ whitened)
        nll = .5 * (2 * np.sum(np.log(np.diag(chol))) + energy + 2 * np.log(2 * np.pi))
        output.append(dict(
            source_row=packet['source_row'], subject=packet['landmark_subject'],
            predicted_range=distance, bearing_innovation=float(innovation[0]),
            range_innovation=float(innovation[1]), nll=float(nll), energy=energy,
            bearing_variance=float(variance[0]), range_variance=float(variance[1]),
            state_covariance=state_covariance))
    return output


def replay(packets, warmup_s, duration_s, methods=METHODS,
           calibrator=calibrate_robot, robot_ids=None, predictor=predicted_observations):
    if (warmup_s <= 0. or duration_s <= warmup_s or 'fixed' not in methods or
            len(set(methods)) != len(methods) or
            (calibrator is calibrate_robot and not set(methods) <= set(METHODS))):
        raise ValueError('require fixed reference and 0 < warmup < duration')
    robots = {method: {} for method in methods}
    timing = {method: dict(update_s=0., prediction_s=0., calibration_s=0.,
                           max_total_backlog_s=0., finish_s=0., update_ms=[]) for method in methods}
    rows, calibration, poses, pose_times = [], [], [], {}
    first_time, switched, count = None, False, 0
    for _, group in groupby(packets, key=lambda packet: (packet['robot'], packet['timestamp_s'])):
        frame = tuple(group)
        robot_id = frame[0]['robot']
        timestamp = frame[0]['timestamp_s']
        if first_time is None:
            first_time = timestamp
        elapsed = timestamp - first_time
        if elapsed > duration_s:
            break
        if robot_ids is not None and robot_id not in robot_ids:
            continue
        if not switched and elapsed > warmup_s:
            for method in methods:
                if method == 'fixed':
                    continue
                start = time.perf_counter()
                for robot in robots[method].values():
                    fit = calibrator(robots['fixed'][robot.robot], method)
                    fit['training_max_elapsed_s'] = robot.last_time - first_time
                    calibration.append(fit)
                    if not fit['success']:
                        raise RuntimeError(f'calibration failed: {fit}')
                    robot.observation_variance_model = variance_model(fit)
                    if 'odometry_variance_scales' in fit:
                        scales = np.asarray(fit['odometry_variance_scales'], dtype=float)
                        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0.):
                            raise ValueError('odometry variance scales must be positive')
                        robot.odometry_sigma = robots['fixed'][robot.robot].odometry_sigma * np.sqrt(scales)
                cost = time.perf_counter() - start
                timing[method]['calibration_s'] = cost
                timing[method]['finish_s'] = max(timing[method]['finish_s'], elapsed) + cost
            switched = True
        count += len(frame)
        for packet in frame:
            if packet['kind'] == 'pose':
                pose_times[(robot_id, packet['pose_index'])] = timestamp
        for method in methods:
            if robot_id not in robots[method]:
                if switched:
                    raise ValueError('robot has no warmup data; do not silently fit from future')
                robots[method][robot_id] = IncrementalPoint2Robot(robot_id)
            robot = robots[method][robot_id]
            start = time.perf_counter()
            predicted = predictor(robot, frame) if switched else []
            prediction_s = time.perf_counter() - start
            for row in predicted:
                state = row.pop('state_covariance')
                rows.append(dict(method=method, robot=robot_id, elapsed_s=elapsed,
                                 state_bearing_variance=float(state[0, 0]),
                                 state_range_variance=float(state[1, 1]),
                                 state_cross_covariance=float(state[0, 1]), **row))
            start = time.perf_counter()
            robot.push_batch(frame)
            pose = robot.graph.solver.calculateEstimatePose2(gtsam.symbol('x', robot.pose_index))
            update_s = time.perf_counter() - start
            poses.append(dict(method=method, robot=robot_id, kind='pose',
                              index=robot.pose_index, timestamp_s=timestamp,
                              x=pose.x(), y=pose.y(), yaw=pose.theta()))
            t = timing[method]
            t['update_s'] += update_s
            t['prediction_s'] += prediction_s
            t['update_ms'].append(1000 * update_s)
            t['finish_s'] = max(t['finish_s'], elapsed) + prediction_s + update_s
            t['max_total_backlog_s'] = max(t['max_total_backlog_s'], t['finish_s'] - elapsed)
    if not switched:
        raise ValueError('no future validation events after warmup')
    summaries = []
    for method in methods:
        matched = [row for row in rows if row['method'] == method]
        t = timing[method]
        times = t.pop('update_ms')
        t.pop('finish_s')
        summaries.append(dict(method=method, predictions=len(matched),
                              mean_prequential_nll=float(np.mean([r['nll'] for r in matched])),
                              mean_predictive_energy=float(np.mean([r['energy'] for r in matched])),
                              update_p99_ms=float(np.quantile(times, .99)), **t))
    final = {method: final_estimate_rows(local, pose_times, 0) for method, local in robots.items()}
    return dict(calibration=calibration, predictions=rows, online_poses=poses,
                final_estimates=final, summary=summaries, packet_count=count,
                start_time_s=first_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=float, default=30.)
    parser.add_argument('--duration', type=float, default=120.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    source = Path(__file__).resolve()
    root = source.parents[1]
    sources = [source, root / 'scripts/dranppr_profiled_noise.py',
               root / 'scripts/dranppr_incremental.py',
               root / 'scripts/dranppr_conditional_noise.py',
               root / 'scripts/dranppr_linear_evidence.py',
               source.parent / 'audit_residual_evidence.py']
    hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    metadata = json.loads((args.stream / 'metadata.json').read_text())
    events = args.stream / 'events.jsonl'
    if sha256(events) != metadata['event_sha256']:
        raise ValueError('event stream digest mismatch')
    with events.open() as stream:
        result = replay((json.loads(line) for line in stream), args.warmup, args.duration)
    for name in ('predictions', 'online_poses', 'summary'):
        write_csv(args.output / f'{name}.csv', result[name])
    (args.output / 'calibration.json').write_text(json.dumps(result['calibration'], indent=2) + '\n')
    dataset = metadata['dataset']
    for method, rows in result['final_estimates'].items():
        directory = args.output / method
        directory.mkdir()
        for row in rows:
            row['dataset'] = dataset
        write_csv(directory / f'dataset{dataset}_final_estimates.csv', rows)
        online = [row for row in result['online_poses'] if row['method'] == method]
        write_csv(directory / f'dataset{dataset}_online_estimates.csv', online)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes=hashes, dataset=dataset,
        event_sha256=metadata['event_sha256'], packet_count=result['packet_count'],
        warmup_s=args.warmup, duration_s=args.duration, truth_used=False,
        relation_edits=False, distributed=False, old_factor_reweighting=False,
        calibration='past-only batch-refined fixed linearization; then frozen',
        prediction='full parent-pose/landmark cross covariance plus odometry propagation',
        first_observation='nominal covariance; not prequentially scored',
        timing='all dense prediction and calibration work included in serial backlog'), indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
