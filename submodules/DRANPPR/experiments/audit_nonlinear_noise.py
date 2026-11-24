"""Compare fixed covariance candidates under linear/nonlinear factor/frame deletion.

This is a retrospective warmup audit, not an online optimizer or a new fit.
Complete observation frames are deleted, while odometry remains available.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import gtsam
import numpy as np
from scipy.linalg import qr, solve_triangular

from audit_online_lower import replay_fixed_calibration
from audit_online_noise import METHODS, variance_model
from audit_residual_evidence import linearize, sha256, write_csv
from dranppr_profiled_noise import DeletedLinearLikelihood


def observation_groups(items, unit):
    if unit not in ('factor', 'frame'):
        raise ValueError('deletion unit must be factor or frame')
    groups = {}
    for index, (factor_id, factor) in enumerate(items):
        if factor_id[0] == 'observation':
            key = index if unit == 'factor' else factor.keys()[0]
            groups.setdefault(key, []).append(index)
    return list(groups.values())


def reweight_items(items, variance):
    result = []
    for index, (factor_id, factor) in enumerate(items):
        if index in variance:
            if factor_id[0] != 'observation':
                raise ValueError('only observation factors may be changed')
            v = np.asarray(variance[index])
            if v.shape != (2,) or not np.isfinite(v).all() or np.any(v <= 0.):
                raise ValueError('invalid observation variance')
            factor = gtsam.BearingRangeFactor2D(
                *factor.keys(), factor.measured().bearing(), factor.measured().range(),
                gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(v)))
        result.append((factor_id, factor))
    return result


def deleted_prediction(a, residual, held, sigmas, solve_linear):
    keep = np.ones(len(a), dtype=bool)
    keep[held] = False
    q, r = qr(a[keep], mode='economic', check_finite=False)
    tolerance = max(a.shape) * np.finfo(float).eps * np.linalg.norm(r, np.inf)
    if r.shape[0] < a.shape[1] or np.any(np.abs(np.diag(r)) <= tolerance):
        raise ValueError('deleted model is not observable')
    innovation = -residual[held]
    if solve_linear:
        step = -solve_triangular(r, q.T @ residual[keep], check_finite=False)
        innovation = innovation - a[held] @ step
    response = solve_triangular(r.T, a[held].T, lower=True, check_finite=False)
    covariance = np.eye(len(held)) + response.T @ response
    return sigmas * innovation, sigmas[:, None] * covariance * sigmas[None, :]


def gaussian_nll(innovation, covariance):
    chol = np.linalg.cholesky(covariance)
    white = solve_triangular(chol, innovation, lower=True, check_finite=False)
    return float(.5 * (white @ white + 2 * np.log(np.diag(chol)).sum() +
                       len(white) * np.log(2 * np.pi)))


def graph_from_items(items, excluded=(), gauge=None):
    graph = gtsam.NonlinearFactorGraph()
    for i, (_, factor) in enumerate(items):
        if i not in excluded:
            graph.push_back(factor)
    if gauge is not None:
        graph.push_back(gauge)
    return graph


def optimize(graph, initial):
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-10)
    params.setAbsoluteErrorTol(1e-10)
    solver = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    start = time.perf_counter()
    values = solver.optimize()
    return values, solver.iterations(), time.perf_counter() - start


def audit_robot(robot, calibration):
    nominal, _, _ = optimize(robot.graph.factor_graph(), robot.graph.estimate())
    items = robot.physical_factor_items()
    a, residual, keys = linearize(robot.physical_graph(), nominal)
    dimensions = [3 if gtsam.Symbol(key).chr() == ord('x') else 2 for key in keys]
    free_columns, cursor = [], 0
    for key, dimension in zip(keys, dimensions):
        if key != gtsam.symbol('x', 0):
            free_columns.extend(range(cursor, cursor + dimension))
        cursor += dimension
    sizes = [factor.dim() for _, factor in items]
    offsets = np.cumsum([0, *sizes])
    block_rows = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(items))]
    observations = [i for i, (key, _) in enumerate(items) if key[0] == 'observation']
    predictor = DeletedLinearLikelihood(a, -residual, sizes, observations)
    distances = {i: items[i][1].measured().range() - .1 * innovation[1]
                 for i, innovation, _ in predictor.predictions(np.ones(len(residual)))}
    eligible, omitted = [], []
    for group in observation_groups(items, 'frame'):
        held = np.concatenate([block_rows[i] for i in group])
        try:
            if not all(i in distances for i in group):
                raise ValueError('unobservable factor')
            deleted_prediction(a[:, free_columns], residual, held, np.ones(len(held)), True)
            eligible.append(group)
        except ValueError:
            omitted.append(group)
    if not eligible:
        raise ValueError('no observable frames')
    accepted = {i for group in eligible for i in group}
    groups = {'frame': eligible, 'factor': [[i] for i in sorted(accepted)]}
    all_rows, folds, summary = [], [], []
    for method in METHODS:
        variances = ({} if method == 'fixed' else
                     {i: variance_model(calibration[method])(d) for i, d in distances.items()})
        weighted = reweight_items(items, variances)
        sigmas = np.concatenate([factor.noiseModel().sigmas() for _, factor in weighted])
        full = graph_from_items(weighted)
        initial_a, initial_r, initial_keys = linearize(full, nominal)
        assert initial_keys == keys
        for unit, blocks in groups.items():
            for block in blocks:
                held = np.concatenate([block_rows[i] for i in block])
                retained = graph_from_items(weighted, block, robot.graph.factors['gauge'])
                refined, iterations, solve_s = optimize(retained, nominal)
                for response in ('linear', 'nonlinear'):
                    if response == 'linear':
                        jacobian, error = initial_a, initial_r
                    else:
                        jacobian, error, new_keys = linearize(full, refined)
                        assert new_keys == keys
                    innovation, covariance = deleted_prediction(
                        jacobian[:, free_columns], error, held, sigmas[held], response == 'linear')
                    mask = np.ones(len(error), dtype=bool)
                    mask[held] = False
                    gradient = jacobian[mask][:, free_columns].T @ error[mask]
                    joint_nll = gaussian_nll(innovation, covariance)
                    fold = dict(method=method, unit=unit, response=response,
                                first_source_row=items[block[0]][0][1], observations=len(block),
                                joint_nll=joint_nll,
                                linearization_cost=float(.5 * error[mask] @ error[mask]),
                                linearization_gradient_norm=float(np.linalg.norm(gradient)),
                                iterations=iterations if response == 'nonlinear' else 0,
                                solve_s=solve_s if response == 'nonlinear' else 0.)
                    folds.append(fold)
                    for j, i in enumerate(block):
                        selection = slice(2*j, 2*j+2)
                        e, c = innovation[selection], covariance[selection, selection]
                        all_rows.append(dict(
                            method=method, unit=unit, response=response,
                            source_row=items[i][0][1], nll=gaussian_nll(e, c),
                            bearing_innovation=float(e[0]), range_innovation=float(e[1]),
                            bearing_variance=float(c[0, 0]), range_variance=float(c[1, 1]),
                            cross_covariance=float(c[0, 1])))
            for response in ('linear', 'nonlinear'):
                rows = [r for r in all_rows if (r['method'], r['unit'], r['response']) ==
                        (method, unit, response)]
                f = [r for r in folds if (r['method'], r['unit'], r['response']) ==
                     (method, unit, response)]
                result = dict(method=method, unit=unit, response=response,
                              observations=len(rows), folds=len(f),
                              mean_marginal_nll=float(np.mean([r['nll'] for r in rows])),
                              joint_nll_per_observation=sum(r['joint_nll'] for r in f) / len(rows),
                              total_solve_s=sum(r['solve_s'] for r in f),
                              maximum_iterations=max(r['iterations'] for r in f),
                              iteration_limit_folds=sum(r['iterations'] >= 100 for r in f))
                summary.append(result)
                print(json.dumps(result), flush=True)
    return all_rows, folds, summary, dict(
        training_observations=len(observations), identifiable_factors=len(distances),
        scored_observations=len(accepted), scored_frames=len(eligible),
        omitted_frames=[[items[i][0][1] for i in group] for group in omitted])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    producer = json.loads((args.run / 'manifest.json').read_text())
    events = args.stream / 'events.jsonl'
    if sha256(events) != producer['event_sha256']:
        raise ValueError('event digest mismatch')
    with events.open() as stream:
        robots, _ = replay_fixed_calibration(
            (json.loads(line) for line in stream), [], producer['warmup_s'],
            producer['warmup_s'], ('fixed',))
    fits = json.loads((args.run / 'calibration.json').read_text())
    fits = {row['method']: row for row in fits if row['robot'] == args.robot}
    start = time.perf_counter()
    rows, folds, summary, counts = audit_robot(robots['fixed'][args.robot], fits)
    for name, table in [('predictions', rows), ('folds', folds), ('summary', summary)]:
        write_csv(args.output / f'{name}.csv', table)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'experiments/audit_online_lower.py',
               root / 'experiments/audit_online_noise.py', root / 'experiments/audit_residual_evidence.py',
               root / 'scripts/dranppr_incremental.py', root / 'scripts/dranppr_profiled_noise.py']
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in sources},
        producer_manifest_sha256=sha256(args.run / 'manifest.json'),
        calibration_sha256=sha256(args.run / 'calibration.json'),
        event_sha256=producer['event_sha256'], robot=args.robot, **counts,
        elapsed_s=time.perf_counter()-start, truth_used=False,
        calibration='no new fit; existing four covariance candidates',
        conditional_covariates='frozen nominal factor-deleted predictions, identical across deletion units',
        exclusion='all observation factors in sensor frame; keep odometry',
        uncertainty='Gauss-Newton covariance at explicit nonlinear deleted solution; not exact posterior',
        matched='same observable observations in every arm; joint and marginal NLL reported separately',
        interpretation='retrospective warmup diagnostic, not independent future validation'), indent=2) + '\n')


if __name__ == '__main__':
    main()
