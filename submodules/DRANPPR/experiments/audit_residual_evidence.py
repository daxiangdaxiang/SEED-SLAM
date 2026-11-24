"""Controlled residual-cause audit and native independent incremental reference.

Not an end-to-end DRANPPR performance comparison. Dense SVD and batch refinement
are diagnostics excluded from the online update timing and never fed back.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import resource
import sys
import time

import gtsam
import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_incremental import IncrementalPoint2Robot
from dranppr_conditional_noise import (
    fit_constant_measurement_variance,
    fit_range_conditional_variance,
    gaussian_mean_nll,
)
from dranppr_linear_evidence import (
    LinearResidualEvidence,
    deleted_group_scale_evidence,
    factor_deleted_scores,
    factor_excluded_prediction,
    profile_deleted_group_scale,
)


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def linear_graph(seed):
    """Two translating robots, known orientations, common object positions."""
    rng = np.random.default_rng(seed)
    n = 12
    nodes = 2 * n + 4
    truth = np.vstack([np.column_stack([np.linspace(0, 4, n), np.zeros(n)]),
                       np.column_stack([np.linspace(4, 0, n), np.ones(n)]),
                       [[0., 2.], [1.5, -1.], [3., 2.], [4.5, -1.]]])
    rows, groups = [], []
    for robot in range(2):
        for t in range(n):
            pose = robot * n + t
            edges = [(pose, 2*n + t % 4, .1), (pose, 2*n + (t+1) % 4, .1)]
            edge_groups = [f'object_sensor_{robot}', f'object_sensor_{robot}']
            if t:
                edges.append((pose - 1, pose, .02))
                edge_groups.append(f'odometry_{robot}')
            for (i, j, sigma), group in zip(edges, edge_groups):
                row = np.zeros(nodes)
                row[i], row[j] = -1. / sigma, 1. / sigma
                rows.append(row[1:])
                groups.append(group)
    a = np.kron(np.asarray(rows), np.eye(2))
    x = truth[1:].ravel()
    y = a @ x + rng.normal(size=a.shape[0])
    return a, x, y, groups


def synthetic_audit(seeds):
    records = []
    for seed in range(seeds):
        a, truth, y, _ = linear_graph(seed)
        evidence = LinearResidualEvidence(a)
        rng = np.random.default_rng(10000 + seed)
        for error in (.0, .1, 1.):
            initial = truth + error * rng.normal(size=truth.shape)
            for budget in (0, 1, 5, 20, 100):
                state = initial
                if budget:
                    state, _ = cg(a.T @ a, a.T @ y, x0=initial,
                                  maxiter=budget, tol=1e-12)
                result = evidence.evaluate(a @ state - y)
                records.append(dict(case='unchanged_data', seed=seed,
                                    initial_error_m=error, iterations=budget,
                                    fitted_ml_scale2=result.total_energy / len(y),
                                    state_rmse=float(np.sqrt(np.mean((state-truth)**2))),
                                    **asdict(result)))
        # A changed observation is deliberately not called native sensor noise.
        for case, changed in (
                ('one_corrupt_observation', y + np.eye(1, len(y), 0)[0] * 20.),
                ('unidentifiable_state_equivalent_bias', y + a @ np.ones(a.shape[1]))):
            result = evidence.evaluate(a @ truth - changed)
            records.append(dict(case=case, seed=seed, initial_error_m=0., iterations=0,
                                fitted_ml_scale2=result.total_energy / len(y),
                                state_rmse=0., **asdict(result)))
    return records


def predictive_scores(a, y, groups, case, seed, target_block=None):
    if a.shape[0] != 2 * len(groups):
        raise ValueError('predictive audit expects one two-dimensional block per group label')
    records = []
    for block, group in enumerate(groups):
        heldout = [2 * block, 2 * block + 1]
        result = factor_excluded_prediction(a, y, heldout)
        records.append(dict(
            case=case, seed=seed, block=block, group=group,
            target=block == target_block, identifiable=result.identifiable,
            normalized_energy=result.normalized_energy,
            energy_per_dimension=(result.normalized_energy / 2.
                                  if result.normalized_energy is not None else None),
            innovation_norm=float(np.linalg.norm(result.innovation)),
            predictive_trace=float(np.trace(result.predictive_covariance)),
            training_rank=result.training_rank,
            unobservable_norm=result.unobservable_norm))
    return records


def synthetic_predictive_audit(seeds):
    records = []
    for seed in range(seeds):
        a, _, y, groups = linear_graph(seed)
        records.extend(predictive_scores(a, y, groups, 'nominal', seed))
        # A state-equivalent change must remain observationally equivalent.
        records.extend(predictive_scores(
            a, y + a @ np.ones(a.shape[1]), groups,
            'state_equivalent_bias', seed))
        rng = np.random.default_rng(20000 + seed)
        degraded = y.copy()
        sensor_zero = [i for i, group in enumerate(groups)
                       if group == 'object_sensor_0']
        for block in sensor_zero:
            degraded[2*block:2*block+2] += np.sqrt(8.) * rng.normal(size=2)
        records.extend(predictive_scores(
            a, degraded, groups, 'group_sensor_degradation', seed))
        corrupted = y.copy()
        target = sensor_zero[len(sensor_zero) // 2]
        corrupted[2*target:2*target+2] += np.array([15., -15.])
        records.extend(predictive_scores(
            a, corrupted, groups, 'one_wrong_relationship', seed, target))
    return records


def linearize(graph, values):
    ordering = gtsam.Ordering()
    keys = list(graph.keyVector())
    if not keys:
        raise ValueError('physical graph has no variables')
    for key in keys:
        ordering.push_back(key)
    a, b = graph.linearize(values).jacobian(ordering)
    return a, -b, keys


def perturb(values, keys, amount, rng):
    delta = gtsam.VectorValues()
    for key in keys:
        dim = 3 if gtsam.Symbol(key).chr() == ord('x') else 2
        step = amount * rng.normal(size=dim)
        if dim == 3:
            step[2] *= .2
        delta.insert(key, step)
    return values.retract(delta)


def deleted_group_calibration_rows(factor_items, deleted, dataset, robot,
                                   elapsed_s):
    """Summarize factor-excluded scale evidence by physical factor type."""
    if len(factor_items) != len(deleted):
        raise ValueError('factor items and deleted evidence must align')
    groups = {}
    for (factor_id, _), evidence in zip(factor_items, deleted):
        kind, _ = factor_id
        if evidence.identifiable:
            groups.setdefault((kind, evidence.dimension), []).append(evidence)
    rows = []
    for (kind, dimension), blocks in sorted(groups.items()):
        if len(blocks) < 2:
            continue
        evidence = deleted_group_scale_evidence(blocks)
        cross = evidence.cross_fitted
        rows.append(dict(
            dataset=dataset, robot=robot, elapsed_s=elapsed_s,
            factor_kind=kind, factor_count=len(blocks), dimension=dimension,
            inclusive_scale=evidence.inclusive.scale,
            inclusive_nll_per_dimension=(
                evidence.inclusive.negative_log_likelihood /
                (len(blocks) * dimension)),
            inclusive_at_zero=evidence.inclusive.at_zero_boundary,
            median_excluded_scale=float(np.median(cross.excluded_scales)),
            q10_excluded_scale=float(np.quantile(cross.excluded_scales, .1)),
            q90_excluded_scale=float(np.quantile(cross.excluded_scales, .9)),
            median_cross_fitted_energy_per_dimension=float(
                np.median(cross.normalized_energy) / dimension),
            median_cross_fitted_nll_per_dimension=float(
                np.median(cross.negative_log_likelihood) / dimension),
            state_covariance_frozen_at_base_scale=True))
    return rows


def deleted_observation_model_rows(factor_items, deleted, dataset, robot,
                                   elapsed_s, bearing_sigma=.05,
                                   range_sigma=.1):
    """Fit marginal bearing and range models from deleted predictions."""
    if len(factor_items) != len(deleted):
        raise ValueError('factor items and deleted evidence must align')
    innovations, state_variances, distances = [], [], []
    sigma = np.asarray([bearing_sigma, range_sigma], dtype=float)
    for (factor_id, factor), evidence in zip(factor_items, deleted):
        if factor_id[0] != 'observation' or not evidence.identifiable:
            continue
        if evidence.dimension != 2:
            raise ValueError('bearing-range factor must have dimension two')
        innovations.append(sigma * evidence.innovation)
        diagonal = np.diag(evidence.predictive_covariance) - 1.
        state_variances.append(np.maximum(0., sigma**2 * diagonal))
        distances.append(float(factor.measured().range()))
    if len(innovations) < 2:
        return []
    innovations = np.asarray(innovations)
    state_variances = np.asarray(state_variances)
    distances = np.asarray(distances)
    bearing = fit_constant_measurement_variance(
        innovations[:, 0], state_variances[:, 0])
    range_constant = fit_constant_measurement_variance(
        innovations[:, 1], state_variances[:, 1])
    range_conditional = fit_range_conditional_variance(
        innovations[:, 1], distances, state_variances[:, 1])
    return [dict(
        dataset=dataset, robot=robot, elapsed_s=elapsed_s,
        factor_count=len(innovations),
        min_range_m=float(np.min(distances)),
        median_range_m=float(np.median(distances)),
        max_range_m=float(np.max(distances)),
        median_abs_bearing_innovation=float(np.median(
            np.abs(innovations[:, 0]))),
        median_abs_range_innovation=float(np.median(
            np.abs(innovations[:, 1]))),
        median_bearing_state_std=float(np.median(
            np.sqrt(state_variances[:, 0]))),
        median_range_state_std=float(np.median(
            np.sqrt(state_variances[:, 1]))),
        bearing_variance=bearing.variance,
        bearing_mean_nll=bearing.mean_nll,
        bearing_nominal_mean_nll=gaussian_mean_nll(
            innovations[:, 0], state_variances[:, 0] + bearing_sigma**2),
        range_constant_variance=range_constant.variance,
        range_constant_mean_nll=range_constant.mean_nll,
        range_intercept_variance=range_conditional.intercept_variance,
        range_distance_squared_coefficient=(
            range_conditional.distance_squared_coefficient),
        range_conditional_mean_nll=range_conditional.mean_nll,
        range_conditional_nll_gain=(range_constant.mean_nll -
                                    range_conditional.mean_nll),
        range_nominal_mean_nll=gaussian_mean_nll(
            innovations[:, 1], state_variances[:, 1] + range_sigma**2),
        fit_success=(bearing.success and range_constant.success and
                     range_conditional.success),
        state_covariance_frozen_at_base_scale=True,
        calibration_inclusive_across_deleted_factors=True)]


def profiled_group_calibration_rows(jacobian, residual, factor_items,
                                    block_sizes, dataset, robot, elapsed_s):
    target = np.asarray([factor_id[0] == 'observation'
                         for factor_id, _ in factor_items], dtype=bool)
    if np.sum(target) < 2:
        return []
    evidence = profile_deleted_group_scale(
        jacobian, -residual, block_sizes, target)
    dimension = sum(size for size, selected in zip(block_sizes, target)
                    if selected)
    return [dict(
        dataset=dataset, robot=robot, elapsed_s=elapsed_s,
        factor_kind='observation', factor_count=evidence.target_factors,
        profiled_scale=evidence.scale,
        profiled_nll_per_dimension=(evidence.negative_log_likelihood /
                                    dimension),
        function_evaluations=evidence.function_evaluations,
        state_response_recomputed_for_each_scale=True)]


def checkpoint_audit(robot, dataset, elapsed_s, rng, profile_group_scale=False):
    physical = robot.physical_graph()
    if physical.size() == 0:
        return [], [], [], [], []
    current = robot.graph.estimate()
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-10)
    params.setAbsoluteErrorTol(1e-10)
    batch = gtsam.LevenbergMarquardtOptimizer(robot.graph.factor_graph(), current, params).optimize()
    records = []
    for variant, values in [('incremental', current), ('batch_refined', batch),
                            ('state_perturbed_01', perturb(batch, list(batch.keys()), .1, rng)),
                            ('state_perturbed_10', perturb(batch, list(batch.keys()), 1., rng))]:
        a, residual, keys = linearize(physical, values)
        evidence = LinearResidualEvidence(a)
        result = evidence.evaluate(residual)
        moved = evidence.evaluate(residual + a @ rng.normal(size=a.shape[1]))
        # Independent dense least squares verifies the projection numerically.
        delta = lstsq(a, -residual, lapack_driver='gelsd')[0]
        reference_energy = float(np.linalg.norm(residual + a @ delta)**2)
        records.append(dict(dataset=dataset, robot=robot.robot, elapsed_s=elapsed_s,
                            variant=variant, factors=physical.size(), variables=len(keys),
                            residual_dim=len(residual), state_dim=a.shape[1],
                            fitted_ml_scale2=result.total_energy / len(residual),
                            invariance_error=abs(moved.contrast_energy - result.contrast_energy),
                            lstsq_error=abs(reference_energy - result.contrast_energy),
                            **asdict(result)))
    a, residual, _ = linearize(physical, batch)
    factor_items = robot.physical_factor_items()
    block_sizes = [factor.dim() for _, factor in factor_items]
    deleted = factor_deleted_scores(a, residual, block_sizes)
    predictive = []
    for factor_index, ((factor_id, _), evidence) in enumerate(zip(factor_items, deleted)):
        kind, source_index = factor_id
        predictive.append(dict(
            dataset=dataset, robot=robot.robot, elapsed_s=elapsed_s,
            factor_index=factor_index, factor_kind=kind, source_index=source_index,
            dimension=evidence.dimension, identifiable=evidence.identifiable,
            normalized_energy=evidence.normalized_energy,
            energy_per_dimension=evidence.energy_per_dimension,
            maximum_leverage=evidence.maximum_leverage,
            minimum_deleted_precision=evidence.minimum_deleted_precision))
    calibration = deleted_group_calibration_rows(
        factor_items, deleted, dataset, robot.robot, elapsed_s)
    observation_models = deleted_observation_model_rows(
        factor_items, deleted, dataset, robot.robot, elapsed_s)
    profiled = (profiled_group_calibration_rows(
        a, residual, factor_items, block_sizes, dataset, robot.robot, elapsed_s)
        if profile_group_scale else [])
    return records, predictive, calibration, observation_models, profiled


def final_estimate_rows(robots, pose_times, dataset):
    rows = []
    for robot_id, robot in sorted(robots.items()):
        values = robot.graph.estimate()
        for key in values.keys():
            symbol = gtsam.Symbol(key)
            index = symbol.index()
            if symbol.chr() == ord('x'):
                pose = values.atPose2(key)
                rows.append(dict(dataset=dataset, robot=robot_id, kind='pose',
                                 index=index, timestamp_s=pose_times[(robot_id, index)],
                                 x=pose.x(), y=pose.y(), yaw=pose.theta()))
            elif symbol.chr() == ord('l'):
                point = values.atPoint2(key)
                rows.append(dict(dataset=dataset, robot=robot_id, kind='landmark',
                                 index=index, timestamp_s='', x=point[0], y=point[1],
                                 yaw=''))
    return rows


def native_audit(stream_dir, duration, checkpoints, output, write_updates=True,
                 wildfire_threshold=1e-3, landmark_ordering='natural',
                 write_final_estimates=True, progress_events=0,
                 atomic_frames=False, profile_group_scale=False):
    metadata = json.loads((stream_dir / 'metadata.json').read_text())
    dataset = metadata['dataset']
    path = stream_dir / 'events.jsonl'
    if sha256(path) != metadata['event_sha256']:
        raise ValueError('event stream digest mismatch')
    (robots, rows, audits, predictive, calibrations, observation_models,
     profiled_calibrations, pose_times) = ({}, [], [], [], [], [], [], {})
    first_time = metadata['first_event_time_s']
    if duration < 0:
        raise ValueError('duration must be nonnegative; zero means the full stream')
    limit = None if duration == 0 else duration
    checkpoints = iter(sorted(c for c in checkpoints
                              if c >= 0 and (limit is None or c <= limit)))
    next_check = next(checkpoints, None)
    rng = np.random.default_rng(dataset)
    finish_time = 0.
    last_elapsed = 0.
    update_times = []
    packet_count = 0
    observation_updates = 0
    maximum_backlog = 0.
    start = time.perf_counter()
    diagnostic_seconds = 0.
    with path.open() as stream:
        packets = (json.loads(line) for line in stream)
        if atomic_frames:
            frames = (tuple(group) for _, group in groupby(
                packets, key=lambda packet: (packet['robot'], packet['timestamp_s'])))
        else:
            frames = ((packet,) for packet in packets)
        next_progress = progress_events if progress_events else None
        for frame in frames:
            packet = frame[0]
            elapsed = packet['timestamp_s'] - first_time
            if limit is not None and elapsed > limit:
                break
            while next_check is not None and elapsed > next_check:
                audit_start = time.perf_counter()
                for robot in robots.values():
                    (evidence_rows, predictive_rows, calibration_rows,
                     observation_model_rows, profiled_rows) = checkpoint_audit(
                         robot, dataset, next_check, rng, profile_group_scale)
                    audits.extend(evidence_rows)
                    predictive.extend(predictive_rows)
                    calibrations.extend(calibration_rows)
                    observation_models.extend(observation_model_rows)
                    profiled_calibrations.extend(profiled_rows)
                diagnostic_seconds += time.perf_counter() - audit_start
                next_check = next(checkpoints, None)
            robot_id = packet['robot']
            if robot_id not in robots:
                robots[robot_id] = IncrementalPoint2Robot(
                    robot_id, wildfire_threshold=wildfire_threshold,
                    landmark_ordering=landmark_ordering)
            robot = robots[robot_id]
            for item in frame:
                if item['kind'] == 'pose':
                    pose_times[(robot_id, item['pose_index'])] = item['timestamp_s']
            tick = time.perf_counter()
            stats = robot.push_batch(frame) if atomic_frames else robot.push(packet)
            # Current-pose output is included; full-trajectory extraction is not.
            pose = robot.graph.solver.calculateEstimatePose2(gtsam.symbol('x', robot.pose_index))
            update_ms = 1000. * (time.perf_counter() - tick)
            if not np.isfinite([pose.x(), pose.y(), pose.theta()]).all():
                raise ValueError('non-finite incremental state')
            finish_time = max(finish_time, elapsed) + update_ms / 1000.
            backlog = finish_time - elapsed
            last_elapsed = elapsed
            update_times.append(update_ms)
            packet_count += len(frame)
            frame_observations = sum(item['kind'] == 'observation' for item in frame)
            observation_updates += frame_observations
            maximum_backlog = max(maximum_backlog, backlog)
            if write_updates:
                rows.append(dict(dataset=dataset, robot=robot_id, elapsed_s=elapsed,
                                 kind=('frame' if atomic_frames else packet['kind']),
                                 packets=len(frame), observations=frame_observations,
                                 update_ms=update_ms,
                                 solver_ms=stats.elapsed_ms, relinearized=stats.relinearized,
                                 reeliminated=stats.reeliminated,
                                 factor_count=robot.graph.factor_count,
                                 pose_index=robot.pose_index, output_x=pose.x(), output_y=pose.y(),
                                 output_yaw=pose.theta(), serial_backlog_s=backlog))
            if next_progress is not None and packet_count >= next_progress:
                print(json.dumps(dict(dataset=dataset, progress_packets=packet_count,
                                      progress_updates=len(update_times), elapsed_s=elapsed)),
                      flush=True)
                while next_progress <= packet_count:
                    next_progress += progress_events
    if next_check is not None:
        audit_start = time.perf_counter()
        for robot in robots.values():
            (evidence_rows, predictive_rows, calibration_rows,
             observation_model_rows, profiled_rows) = checkpoint_audit(
                 robot, dataset, last_elapsed, rng, profile_group_scale)
            audits.extend(evidence_rows)
            predictive.extend(predictive_rows)
            calibrations.extend(calibration_rows)
            observation_models.extend(observation_model_rows)
            profiled_calibrations.extend(profiled_rows)
        diagnostic_seconds += time.perf_counter() - audit_start
    wall = time.perf_counter() - start
    if write_updates and rows:
        write_csv(output / f'dataset{dataset}_updates.csv', rows)
    if audits:
        write_csv(output / f'dataset{dataset}_evidence.csv', audits)
    if predictive:
        write_csv(output / f'dataset{dataset}_predictive.csv', predictive)
    if calibrations:
        write_csv(output / f'dataset{dataset}_calibration.csv', calibrations)
    if observation_models:
        write_csv(output / f'dataset{dataset}_observation_model.csv',
                  observation_models)
    if profiled_calibrations:
        write_csv(output / f'dataset{dataset}_profiled_calibration.csv',
                  profiled_calibrations)
    estimate_start = time.perf_counter()
    estimate_rows = final_estimate_rows(robots, pose_times, dataset)
    estimate_seconds = time.perf_counter() - estimate_start
    if write_final_estimates:
        write_csv(output / f'dataset{dataset}_final_estimates.csv', estimate_rows)
    if not update_times:
        raise ValueError('duration selected no events')
    times = np.asarray(update_times)
    summary = dict(dataset=dataset, packets=packet_count, updates=len(update_times),
        observation_updates=observation_updates, duration_s=last_elapsed,
        p50_ms=float(np.median(times)), p95_ms=float(np.quantile(times, .95)),
        p99_ms=float(np.quantile(times, .99)), max_ms=float(max(times)),
        total_update_s=float(times.sum()/1000.), replay_wall_s=wall,
        diagnostic_s=diagnostic_seconds,
        calibration_rows=len(calibrations),
        observation_model_rows=len(observation_models),
        profiled_calibration_rows=len(profiled_calibrations),
        final_estimate_rows=len(estimate_rows),
        final_estimate_extraction_s=estimate_seconds,
        process_peak_rss_mb=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                             1024.),
        max_serial_backlog_s=maximum_backlog,
        event_sha256=metadata['event_sha256'],
        wildfire_threshold=wildfire_threshold,
        landmark_ordering=landmark_ordering,
        atomic_frames=atomic_frames,
        method='independent GTSAM iSAM2; NOT DRANPPR distributed optimizer')
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--streams', type=Path, default=Path('/tmp/dranppr_causal_streams_v1'))
    parser.add_argument('--datasets', type=int, nargs='*', default=[1, 2, 3, 4, 9])
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--checkpoints', type=float, nargs='*', default=[10., 30., 60.])
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--skip-controlled', action='store_true',
                        help='skip synthetic audits for a native timing-only replay')
    parser.add_argument('--summary-only', action='store_true',
                        help='do not retain per-event update CSV files')
    parser.add_argument('--wildfire-threshold', type=float, default=1e-3,
                        help='iSAM2 incremental back-substitution threshold')
    parser.add_argument('--constrain-landmarks-last', action='store_true',
                        help='legacy negative ablation; equivalent to --landmark-ordering all_last')
    parser.add_argument('--landmark-ordering', choices=['natural', 'new_last', 'all_last'],
                        default='natural',
                        help='causal elimination structure for static landmarks')
    parser.add_argument('--no-final-estimates', action='store_true',
                        help='do not retain final trajectory and landmark estimates')
    parser.add_argument('--progress-events', type=int, default=0)
    parser.add_argument('--atomic-frames', action='store_true',
                        help='insert all factors at one robot/timestamp in one transaction')
    parser.add_argument('--profile-group-scale', action='store_true',
                        help='recompute the exact linear state response for each tested group scale')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    if args.constrain_landmarks_last and args.landmark_ordering != 'natural':
        raise ValueError('legacy and explicit landmark ordering options conflict')
    landmark_ordering = ('all_last' if args.constrain_landmarks_last
                         else args.landmark_ordering)
    producer = Path(__file__).resolve()
    scripts = producer.parents[1] / 'scripts'
    source_paths = (producer, scripts / 'dranppr_linear_evidence.py',
                    scripts / 'dranppr_incremental.py',
                    scripts / 'dranppr_conditional_noise.py')
    source_hashes = {str(path.relative_to(producer.parents[1])): sha256(path)
                     for path in source_paths}
    if not args.skip_controlled:
        write_csv(args.output / 'synthetic.csv', synthetic_audit(args.seeds))
        write_csv(args.output / 'predictive.csv', synthetic_predictive_audit(args.seeds))
    summaries = [native_audit(args.streams / f'dataset{d}', args.duration,
                              args.checkpoints, args.output,
                               write_updates=not args.summary_only,
                               wildfire_threshold=args.wildfire_threshold,
                               landmark_ordering=landmark_ordering,
                              write_final_estimates=not args.no_final_estimates,
                              progress_events=args.progress_events,
                              atomic_frames=args.atomic_frames,
                              profile_group_scale=args.profile_group_scale)
                 for d in args.datasets]
    provenance = dict(gtsam_module=gtsam.__file__, numpy=np.__version__, scipy=scipy.__version__,
                      source_hashes=source_hashes,
                      initialization='independent local fixed gauges, no inter-robot registration',
                      covariance='polar sigma=(bearing=.05 rad, range=.1 m); odometry sigma=(.1,.1,.05) per sqrt(second)',
                      adaptation=('none in solver; contrast and factor-deleted '
                                  'group scale are diagnostics only'),
                      calibration=(
                          'factor-deleted local linear prediction; frozen-state '
                          'and optional exact profiled state-response diagnostics'),
                      timing='single-process serial execution; excludes dense audits, includes current-pose output',
                      controlled_audits=not args.skip_controlled,
                      retained_per_event_updates=not args.summary_only,
                      wildfire_threshold=args.wildfire_threshold,
                      landmark_ordering=landmark_ordering,
                      atomic_frames=args.atomic_frames,
                      profile_group_scale=args.profile_group_scale,
                      retained_final_estimates=not args.no_final_estimates,
                      truth_used=False, native_noise_injected=False,
                      argv=sys.argv, summaries=summaries)
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
