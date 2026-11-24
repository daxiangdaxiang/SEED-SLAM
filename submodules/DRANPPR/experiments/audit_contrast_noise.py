"""Diagnose odometry/observation variance confounding with classical REML.

No ground truth, association change, or online deployment occurs here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from audit_nonlinear_noise import optimize
from audit_online_lower import replay_fixed_calibration
from audit_residual_evidence import linearize, sha256, write_csv
from dranppr_profiled_noise import DeletedLinearLikelihood
from dranppr_variance_components import ContrastVarianceModel


def noise_components(items, distances, diagonal_odometry=False):
    scale = float(np.median(list(distances.values())))
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError('positive predicted distances are required')
    size = sum(factor.dim() for _, factor in items)
    odometry_dimension = 3 if diagonal_odometry else 1
    components, base, cursor = np.zeros((size, odometry_dimension+3)), np.ones(size), 0
    for index, (key, factor) in enumerate(items):
        rows = slice(cursor, cursor + factor.dim())
        if key[0] == 'odometry':
            base[rows] = 0.
            if diagonal_odometry:
                components[rows, :3] = np.eye(3)
            else:
                components[rows, 0] = 1.
        elif index in distances:
            base[rows] = 0.
            components[cursor, odometry_dimension] = 1.
            components[cursor + 1, odometry_dimension+1:] = [1., (distances[index]/scale)**2]
        cursor += factor.dim()
    return components, base, scale


def calibrate_contrast(robot, method):
    values, _, _ = optimize(robot.graph.factor_graph(), robot.graph.estimate())
    a, residual, _ = linearize(robot.physical_graph(), values)
    items = robot.physical_factor_items()
    sizes = [factor.dim() for _, factor in items]
    obs = [i for i, (key, _) in enumerate(items) if key[0] == 'observation']
    predictor = DeletedLinearLikelihood(a, -residual, sizes, obs)
    distances = {i: items[i][1].measured().range() - .1*e[1]
                 for i, e, _ in predictor.predictions(np.ones(len(residual)))}
    components, base, scale = noise_components(items, distances, method == 'reml_diagonal_q')
    if method == 'reml_fixed_q':
        model = ContrastVarianceModel(a, -residual, components[:, 1:], base + components[:, 0])
        result = model.fit(np.log([1., .5, .5]))
        q = np.r_[1., result['components']]
    elif method == 'reml_joint_q':
        model = ContrastVarianceModel(a, -residual, components, base)
        result = model.fit(np.log([1., 1., .5, .5]))
        q = np.asarray(result['components'])
    elif method == 'reml_diagonal_q':
        model = ContrastVarianceModel(a, -residual, components, base)
        result = model.fit(np.log([1., 1., 1., 1., .5, .5]))
        q = np.asarray(result['components'])
    else:
        raise ValueError('unknown contrast calibration')
    odometry = q[:3] if method == 'reml_diagonal_q' else np.full(3, q[0])
    result.update(method=method, robot=robot.robot, odometry_variance_scales=odometry.tolist(),
                  bearing_variance=float(.05**2*q[-3]),
                  range_intercept_variance=float(.1**2*q[-2]),
                  range_distance_squared_coefficient=float(.1**2*q[-1]/scale**2),
                  training_observations=len(obs), scored_observations=len(distances),
                  fixed_linearization=True)
    return result, (a, residual, components, base, scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robots', type=int, nargs='+', default=[3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    source = json.loads((args.run / 'manifest.json').read_text())
    events = args.stream / 'events.jsonl'
    if sha256(events) != source['event_sha256']:
        raise ValueError('event digest mismatch')
    with events.open() as stream:
        robots, _ = replay_fixed_calibration((json.loads(line) for line in stream), [],
                    source['warmup_s'], source['warmup_s'], ('fixed',))
    calibration = json.loads((args.run / 'calibration.json').read_text())
    fits, scores = [], []
    start = time.perf_counter()
    for robot_id in args.robots:
        robot = robots['fixed'][robot_id]
        local_fits = []
        for method in ('reml_fixed_q', 'reml_joint_q', 'reml_diagonal_q'):
            fit, problem = calibrate_contrast(robot, method)
            local_fits.append(fit)
            print(json.dumps(fit), flush=True)
        a, residual, d, base, scale = problem
        model = ContrastVarianceModel(a, -residual, d, base)
        candidates = [dict(method='fixed', robot=robot_id, bearing_variance=.05**2,
                           range_intercept_variance=.1**2, range_distance_squared_coefficient=0.)]
        candidates.extend(row for row in calibration if row['robot'] == robot_id)
        candidates.extend(local_fits)
        for candidate in candidates:
            q = np.array([*candidate.get('odometry_variance_scales', [1., 1., 1.]),
                          candidate['bearing_variance']/.05**2,
                          candidate['range_intercept_variance']/.1**2,
                          candidate['range_distance_squared_coefficient']*scale**2/.1**2])
            with np.errstate(divide='ignore'):
                value = model.nll(np.log(q))
            scores.append(dict(robot=robot_id, method=candidate['method'],
                               contrast_nll=value, contrast_dof=len(model.z)))
        fits.extend(local_fits)
    write_csv(args.output / 'scores.csv', scores)
    (args.output / 'calibration.json').write_text(json.dumps(fits, indent=2) + '\n')
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'scripts/dranppr_variance_components.py',
             root/'scripts/dranppr_linear_evidence.py', root/'scripts/dranppr_profiled_noise.py',
             root/'experiments/audit_nonlinear_noise.py', root/'experiments/audit_online_lower.py',
             root/'scripts/dranppr_incremental.py']
    (args.output/'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in paths},
        producer_manifest_sha256=sha256(args.run/'manifest.json'),
        calibration_sha256=sha256(args.run/'calibration.json'), event_sha256=source['event_sha256'],
        truth_used=False, fixed_linearization=True, online=False,
        elapsed_s=time.perf_counter()-start,
        prior_art='Classical error-contrast / restricted likelihood variance-component estimation',
        component_order='one or three local odometry multipliers, bearing, range intercept, range squared-distance'), indent=2)+'\n')


if __name__ == '__main__':
    main()
