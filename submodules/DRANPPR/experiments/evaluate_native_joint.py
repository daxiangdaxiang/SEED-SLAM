"""Posthoc joint state evaluation with ONE component-wide SE(2) alignment."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.materialize import estimate_se2_registration
from benchmarks.mrclam.source import audit_dataset, load_manifest
from experiments.evaluate_incremental_reference import read_rows, evaluate_rows, _float, wrap


def evaluate_joint_rows(rows, truth, label_map):
    all_poses = [row for row in rows if row['kind'] == 'pose']
    poses = [row for row in all_poses if truth.robots[int(row['robot'])].supports(_float(row, 'timestamp_s'))]
    if len(poses) < 2:
        raise ValueError('insufficient ground-truth-supported poses')
    gt = [truth.robots[int(row['robot'])].interpolate(_float(row, 'timestamp_s')) for row in poses]
    estimate = np.array([[_float(row, 'x'), _float(row, 'y')] for row in poses])
    target = np.vstack([item[0] for item in gt])
    alignment, rmse = estimate_se2_registration(estimate, target)
    c, s = math.cos(alignment.yaw), math.sin(alignment.yaw)
    rotation = np.array([[c, -s], [s, c]])
    translation = np.array([alignment.x, alignment.y])
    yaw_error = wrap(np.array([_float(row, 'yaw') for row in poses])+alignment.yaw-
                     np.array([item[1] for item in gt]))
    object_errors = []
    for row in rows:
        if row['kind'] != 'landmark':
            continue
        label = label_map.get(int(row['index']), int(row['index']))
        if label in truth.landmarks:
            point = rotation @ np.array([_float(row, 'x'), _float(row, 'y')])+translation
            object_errors.append(float(np.linalg.norm(point-truth.landmarks[label])))
    return dict(alignment='single_global_SE2_from_all_supported_poses', pose_count=len(poses),
        unsupported_pose_count=len(all_poses)-len(poses), object_copy_count=len(object_errors),
        translation_rmse_m=rmse, yaw_rmse_rad=float(np.sqrt(np.mean(yaw_error**2))),
        object_copy_rmse_m=float(np.sqrt(np.mean(np.square(object_errors)))) if object_errors else None,
        alignment_x=alignment.x, alignment_y=alignment.y, alignment_yaw=alignment.yaw)


def package_rows(package):
    rows = []
    for robot in package['robots']:
        for p in robot['poses']:
            rows.append(dict(robot=robot['id'], kind='pose', index=p['index'], timestamp_s=p['timestamp_s'],
                x=p['value'][0], y=p['value'][1], yaw=p['value'][2]))
        for p in robot['objects']:
            rows.append(dict(robot=robot['id'], kind='landmark', index=p['label'],
                x=p['point'][0], y=p['point'][1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--independent-completed', type=Path, nargs='+', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root/f"benchmarks/mrclam/manifests/dataset{package['provenance']['dataset']}.json")
    audit = audit_dataset(manifest, args.raw)
    truth = load_ground_truth(manifest, audit.dataset_directory)
    label_map = dict(manifest.evaluation.object_ground_truth_label_map)
    result = dict(truth_used_by_estimator=False, truth_used_by_evaluation=True,
        scope='One global alignment; full arrived prefix; object copies counted, not unique identities',
        initial=evaluate_joint_rows(package_rows(package), truth, label_map), runs=[])
    result['independent_online_per_robot_alignment'] = evaluate_rows(
        read_rows(package['provenance']['snapshot']), truth, label_map)[0]
    if args.independent_completed:
        independent = [row for path in args.independent_completed for row in read_rows(path)]
        result['independent_completed_per_robot_alignment'] = evaluate_rows(independent, truth, label_map)[0]
    for path in args.result:
        summary = json.loads((path/'summary.json').read_text())
        summary.pop('trace', None)
        rows = read_rows(path/'final_estimates.csv')
        result['runs'].append(dict(path=str(path), **summary,
            **evaluate_joint_rows(rows, truth, label_map),
            per_robot_alignment_shape_diagnostic=evaluate_rows(rows, truth, label_map)[0]))
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
