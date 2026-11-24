"""Gauge-invariant pair-distance diagnostics conditional on local GN covariance."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.audit_native_checkpoint import sha256
from benchmarks.mrclam.materialize import estimate_se2_registration


def distances(record):
    ids = record['object_ids']
    points = np.asarray(record['means'], dtype=float)
    covariance = np.asarray(record['covariance'], dtype=float)
    n = len(ids)
    if len(set(ids)) != n or points.shape != (n, 2) or not np.isfinite(points).all():
        raise ValueError('invalid object layout')
    if covariance.shape != (2*n, 2*n) or not np.isfinite(covariance).all() or not np.allclose(
            covariance, covariance.T, atol=1e-12, rtol=1e-10):
        raise ValueError('invalid covariance')
    covariance = .5*(covariance+covariance.T)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError('covariance is not positive definite')
    positions = {key: i for i, key in enumerate(ids)}
    rows = []
    for a, b in itertools.combinations(sorted(ids), 2):
        i, j = positions[a], positions[b]
        delta = points[i]-points[j]
        distance = float(np.linalg.norm(delta))
        row = dict(object_a=a, object_b=b, distance_m=distance)
        if distance == 0.:
            rows.append(dict(row, status='undefined_distance_derivative'))
            continue
        direction = delta/distance
        ii = covariance[2*i:2*i+2, 2*i:2*i+2]
        jj = covariance[2*j:2*j+2, 2*j:2*j+2]
        ij = covariance[2*i:2*i+2, 2*j:2*j+2]
        variance = float(direction@(ii+jj-ij-ij.T)@direction)
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError('nonpositive distance covariance')
        rows.append(dict(row, status='defined', variance_m2=variance,
                         block_diagonal_variance_m2=float(direction@(ii+jj)@direction)))
    return rows


def compare_maps(first, second):
    def keyed(record):
        return {(r['object_a'], r['object_b']): r for r in distances(record)}
    left, right = keyed(first), keyed(second)
    rows = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        row = dict(object_a=key[0], object_b=key[1],
                   first_distance_m=a['distance_m'], second_distance_m=b['distance_m'])
        if a['status'] != 'defined' or b['status'] != 'defined':
            rows.append(dict(row, status='undefined_distance_derivative'))
            continue
        difference = a['distance_m']-b['distance_m']
        variance = a['variance_m2']+b['variance_m2']
        diagonal = a['block_diagonal_variance_m2']+b['block_diagonal_variance_m2']
        rows.append(dict(row, status='defined', difference_m=difference,
            combined_variance_m2=variance, combined_std_m=float(np.sqrt(variance)),
            absolute_standardized_difference=abs(difference)/float(np.sqrt(variance)),
            diagonal_only_standardized_difference=abs(difference)/float(np.sqrt(diagonal)),
            diagonal_to_full_variance_ratio=diagonal/variance))
    return rows


def neighbor_object_starts(rows, neighbor):
    """Frozen diagnostic only: preserve every pose/key; replace common object seeds."""
    if len({row['robot'] for row in rows}) != 1:
        raise ValueError('neighbor seed requires one target robot')
    if len({(r['kind'], r['index']) for r in rows}) != len(rows):
        raise ValueError('duplicate target state key')
    distances(neighbor)
    target = {r['index']: np.array([r['x'], r['y']]) for r in rows if r['kind'] == 'landmark'}
    source = dict(zip(neighbor['object_ids'], np.asarray(neighbor['means'])))
    common = sorted(target.keys() & source.keys())
    transform, rms = estimate_se2_registration([source[k] for k in common], [target[k] for k in common])
    c, s = np.cos(transform.yaw), np.sin(transform.yaw)
    rotation, translation = np.array([[c, -s], [s, c]]), np.array([transform.x, transform.y])
    deltas = {k: rotation@source[k]+translation-target[k] for k in common}
    starts = {}
    for label, sign in [('neighbor', 1.), ('opposite', -1.)]:
        result = []
        for row in rows:
            changed = dict(row)
            if row['kind'] == 'landmark' and row['index'] in deltas:
                point = target[row['index']]+sign*deltas[row['index']]
                changed.update(x=float(point[0]), y=float(point[1]))
            result.append(changed)
        starts[label] = result
    return starts, dict(common_objects=common, rigid_fit_rms_m=rms,
        target_from_source=[transform.x, transform.y, transform.yaw],
        object_delta_norm_m=float(np.sqrt(sum(d@d for d in deltas.values()))),
        scope='Only object initial values changed; opposite is equal-norm opposite-direction control, not an estimator option')


def summarize(rows):
    defined = [r for r in rows if r['status'] == 'defined']
    result = dict(pairs=len(rows), defined_pairs=len(defined))
    for field in ['absolute_standardized_difference', 'diagonal_only_standardized_difference',
                  'diagonal_to_full_variance_ratio', 'combined_std_m']:
        values = [r[field] for r in defined]
        result[field] = dict(median=float(np.median(values)), maximum=float(np.max(values))) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', action='append', required=True, help='ROBOT=object_covariance.json')
    parser.add_argument('--links', nargs='+', required=True, help='robot:robot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    maps, paths = {}, {}
    for item in args.map:
        label, name = item.split('=', 1)
        robot, path = int(label), Path(name)
        if robot in maps:
            raise ValueError('duplicate robot map')
        maps[robot] = json.loads(path.read_text())
        paths[robot] = path
    results, summaries, seen = [], [], set()
    for link in args.links:
        edge = tuple(sorted(map(int, link.split(':'))))
        if len(edge) != 2 or edge[0] == edge[1] or edge in seen:
            raise ValueError('invalid or repeated link')
        seen.add(edge)
        a, b = edge
        rows = compare_maps(maps[a], maps[b])
        results.extend(dict(robot_a=a, robot_b=b, **row) for row in rows)
        summaries.append(dict(robot_a=a, robot_b=b, **summarize(rows)))
    args.output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(key for row in results for key in row))
    with (args.output/'pairs.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    record = dict(truth_used=False, maps={r: dict(path=str(p), sha256=sha256(p)) for r, p in paths.items()},
        script_sha256=sha256(__file__), edges=summaries,
        scope='Descriptive pair-distance disagreement of independent local estimates, conditional on chosen Gaussian Q/R and one linearization',
        limitations='Pairs are correlated; no independent-trial counts, p-values or fault decisions. GN covariance excludes nonlinear multimodality, parameter uncertainty and persistent bias. Not a distributed solve.')
    (args.output/'summary.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
