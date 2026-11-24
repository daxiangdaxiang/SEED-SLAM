"""Post-run GT evaluation for the independent local incremental reference.

Ground truth is read only here, never by the replay or estimator. Each robot is
aligned independently because the reference has no inter-robot factors. Its
metrics therefore cannot be compared as a distributed global-map result.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.materialize import estimate_se2_registration
from benchmarks.mrclam.source import audit_dataset, load_manifest


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _float(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f'non-finite {field}')
    return value


def evaluate_rows(rows, truth, label_map):
    results = []
    for robot in sorted({int(row['robot']) for row in rows}):
        local = [row for row in rows if int(row['robot']) == robot]
        if robot not in truth.robots:
            raise ValueError(f'missing trajectory for robot {robot}')
        all_poses = sorted((row for row in local if row['kind'] == 'pose'),
                           key=lambda row: int(row['index']))
        series = truth.robots[robot]
        poses = [row for row in all_poses
                 if series.supports(_float(row, 'timestamp_s'))]
        unsupported_pose_count = len(all_poses) - len(poses)
        if len(poses) < 2:
            raise ValueError(f'missing ground-truth-supported trajectory for robot {robot}')
        estimate_xy = np.array([[_float(row, 'x'), _float(row, 'y')]
                                for row in poses])
        timestamps = np.array([_float(row, 'timestamp_s') for row in poses])
        gt = [series.interpolate(t) for t in timestamps]
        truth_xy = np.vstack([item[0] for item in gt])
        truth_yaw = np.array([item[1] for item in gt])
        registration, translation_rmse = estimate_se2_registration(
            estimate_xy, truth_xy)
        c, s = math.cos(registration.yaw), math.sin(registration.yaw)
        rotation = np.array([[c, -s], [s, c]])
        estimate_yaw = np.array([_float(row, 'yaw') for row in poses])
        yaw_error = wrap(estimate_yaw + registration.yaw - truth_yaw)
        landmarks = [row for row in local if row['kind'] == 'landmark']
        object_errors = []
        for row in landmarks:
            label = label_map.get(int(row['index']), int(row['index']))
            if label not in truth.landmarks:
                continue
            point = rotation @ np.array([_float(row, 'x'), _float(row, 'y')]) + np.array(
                [registration.x, registration.y])
            object_errors.append(float(np.linalg.norm(point - truth.landmarks[label])))
        results.append(dict(
            robot=robot, pose_count=len(poses),
            unsupported_pose_count=unsupported_pose_count,
            object_count=len(object_errors),
            translation_rmse_m=translation_rmse,
            yaw_rmse_rad=float(np.sqrt(np.mean(yaw_error**2))),
            object_rmse_m=(float(np.sqrt(np.mean(np.square(object_errors))))
                           if object_errors else None),
            alignment_x=registration.x, alignment_y=registration.y,
            alignment_yaw=registration.yaw))
    pose_count = sum(row['pose_count'] for row in results)
    unsupported_pose_count = sum(row['unsupported_pose_count'] for row in results)
    object_count = sum(row['object_count'] for row in results)
    aggregate = dict(
        robots=len(results), pose_count=pose_count,
        unsupported_pose_count=unsupported_pose_count,
        object_copy_count=object_count,
        translation_rmse_m=float(np.sqrt(sum(
            row['pose_count'] * row['translation_rmse_m']**2 for row in results) /
            pose_count)),
        yaw_rmse_rad=float(np.sqrt(sum(
            row['pose_count'] * row['yaw_rmse_rad']**2 for row in results) /
            pose_count)),
        object_copy_rmse_m=(float(np.sqrt(sum(
            row['object_count'] * row['object_rmse_m']**2 for row in results
            if row['object_rmse_m'] is not None) / object_count))
                            if object_count else None))
    return aggregate, results


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--manifests', type=Path,
                        default=Path('benchmarks/mrclam/manifests'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    aggregate_rows, robot_rows = [], []
    files = []
    for directory in args.results:
        files.extend(Path(path) for path in glob.glob(
            str(directory / 'dataset*_final_estimates.csv')))
    for path in sorted(files):
        dataset = int(path.name.split('_')[0][7:])
        manifest = load_manifest(args.manifests / f'dataset{dataset}.json')
        source = audit_dataset(manifest, args.raw)
        truth = load_ground_truth(manifest, source.dataset_directory)
        label_map = dict(manifest.evaluation.object_ground_truth_label_map)
        aggregate, robots = evaluate_rows(read_rows(path), truth, label_map)
        aggregate_rows.append(dict(dataset=dataset, **aggregate))
        robot_rows.extend(dict(dataset=dataset, **row) for row in robots)
    if not aggregate_rows:
        raise ValueError('no final estimate files found')
    for name, rows in [('datasets.csv', aggregate_rows), ('robots.csv', robot_rows)]:
        with (args.output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        scope='post-run independently aligned local reference; not distributed DRANPPR',
        truth_used_by_estimator=False, truth_used_by_evaluation=True,
        datasets=len(aggregate_rows), results=aggregate_rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
