"""Post-run evaluation only: truth never enters chronological calibration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_incremental_reference import evaluate_rows, read_rows
from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest


def paired_predictive_summary(rows):
    methods = {}
    for row in rows:
        method = row['method']
        key = (int(row['robot']), int(row['source_row']))
        arm = methods.setdefault(method, {})
        if key in arm:
            raise ValueError('duplicate predictive event')
        if not np.isfinite([float(row['nll']), float(row['energy'])]).all():
            raise ValueError('non-finite prediction')
        arm[key] = row
    fixed = methods['fixed']
    if not fixed:
        raise ValueError('empty prediction comparison')
    summary = {}
    for method, arm in methods.items():
        if arm.keys() != fixed.keys():
            raise ValueError('prediction arms must use the same events')
        differences = [float(row['nll']) - float(fixed[key]['nll']) for key, row in arm.items()]
        summary[method] = dict(predictions=len(arm),
                              mean_prequential_nll=float(np.mean([float(r['nll']) for r in arm.values()])),
                              paired_nll_change=float(np.mean(differences)),
                              mean_predictive_energy=float(np.mean([float(r['energy']) for r in arm.values()])))
    return summary


def noise_failure_diagnostics(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], int(row['robot'])), []).append(row)
    summaries = []
    for (method, robot), group in sorted(groups.items()):
        previous, pairs = {}, []
        for row in sorted(group, key=lambda row: float(row['elapsed_s'])):
            subject = int(row['subject'])
            if subject in previous:
                pairs.append((previous[subject], row))
            previous[subject] = row
        summary = dict(method=method, robot=robot, observations=len(group),
                       mean_nll=float(np.mean([float(row['nll']) for row in group])),
                       mean_predictive_energy=float(np.mean([float(row['energy']) for row in group])),
                       successive_same_identity_pairs=len(pairs),
                       median_pair_time_gap_s=(float(np.median([
                           float(b['elapsed_s']) - float(a['elapsed_s']) for a, b in pairs]))
                           if pairs else None))
        for channel in ('bearing', 'range'):
            pair_values = np.array([[float(a[f'{channel}_innovation']), float(b[f'{channel}_innovation'])]
                                    for a, b in pairs])
            correlation = (float(np.corrcoef(pair_values.T)[0, 1])
                           if len(pairs) > 1 and np.all(np.std(pair_values, axis=0) > 0.) else None)
            summary[f'{channel}_innovation_lag_correlation'] = correlation
            summary[f'mean_{channel}_marginal_energy'] = float(np.mean([
                float(row[f'{channel}_innovation'])**2 / (
                    float(row[f'{channel}_variance']) + float(row[f'state_{channel}_variance']))
                for row in group]))
        summaries.append(summary)
    return summaries


def evaluate_future_estimate(rows, truth, labels, start):
    future = [row for row in rows if row['kind'] == 'landmark' or float(row['timestamp_s']) > start]
    return evaluate_rows(future, truth, labels)


def independent_residual_examples(rows, audit_root, dataset, limit=10,
                                  method='profiled_conditional'):
    fixed = {(int(row['robot']), int(row['source_row'])): row
             for row in rows if row['method'] == 'fixed'}
    candidates = sorted((row for row in rows if row['method'] == method),
                        key=lambda row: float(row['nll']) - float(fixed[(
                            int(row['robot']), int(row['source_row']))]['nll']), reverse=True)
    if not candidates:
        raise ValueError(f'no predictions for failure method: {method}')
    cache, examples = {}, []
    for row in candidates[:limit]:
        robot, source_row = int(row['robot']), int(row['source_row'])
        if robot not in cache:
            with np.load(audit_root / f'd{dataset}_r{robot}_raw.npz') as archive:
                cache[robot] = {int(index): (residual, bool(valid)) for index, residual, valid in zip(
                    archive['raw_row_index'], archive['residual'], archive['valid'])}
        residual, valid = cache[robot][source_row]
        examples.append(dict(robot=robot, source_row=source_row, subject=int(row['subject']),
                             elapsed_s=float(row['elapsed_s']),
                             nll=float(row['nll']), fixed_nll=float(fixed[(robot, source_row)]['nll']),
                             range_innovation=float(row['range_innovation']),
                             bearing_innovation=float(row['bearing_innovation']),
                             gt_supported=valid,
                             gt_range_error=float(residual[0]) if valid else None,
                             gt_bearing_error=float(residual[1]) if valid else None))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-run', type=Path)
    parser.add_argument('--native-residuals', type=Path)
    parser.add_argument('--failure-method', default='profiled_conditional')
    parser.add_argument('--manifests', type=Path, default=Path('benchmarks/mrclam/manifests'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    producer = json.loads((args.run / 'manifest.json').read_text())
    dataset = producer['dataset']
    benchmark = load_manifest(args.manifests / f'dataset{dataset}.json')
    source = audit_dataset(benchmark, args.raw)
    truth = load_ground_truth(benchmark, source.dataset_directory)
    labels = dict(benchmark.evaluation.object_ground_truth_label_map)
    predictions = read_rows(args.run / 'predictions.csv')
    paired = paired_predictive_summary(predictions)
    timing = {row['method']: row for row in read_rows(args.run / 'summary.csv')}
    calibration = json.loads((args.run / 'calibration.json').read_text())
    result, robot_rows = [], []
    for method, comparison in paired.items():
        directory = args.run / method
        final = read_rows(directory / f'dataset{dataset}_final_estimates.csv')
        online = read_rows(directory / f'dataset{dataset}_online_estimates.csv')
        first = producer.get('start_time_s', min(float(row['timestamp_s']) for row in online))
        start = first + producer['warmup_s']
        final_metrics, final_robots = evaluate_future_estimate(final, truth, labels, start)
        online_metrics, online_robots = evaluate_future_estimate(online, truth, labels, start)
        batch_metrics = {}
        if args.batch_run:
            batch = read_rows(args.batch_run / method / f'dataset{dataset}_final_estimates.csv')
            aggregate, robots = evaluate_future_estimate(batch, truth, labels, start)
            robot_rows.extend(dict(method=method, scope='batch', **row) for row in robots)
            batch_metrics = dict(batch_trajectory_rmse_m=aggregate['translation_rmse_m'],
                                 batch_object_copy_rmse_m=aggregate['object_copy_rmse_m'])
        for scope, robots in [('final', final_robots), ('online', online_robots)]:
            robot_rows.extend(dict(method=method, scope=scope, **row) for row in robots)
        t = timing[method]
        fits = [row for row in calibration if row['method'] == method]
        result.append(dict(
            dataset=dataset, method=method, **comparison,
            final_trajectory_rmse_m=final_metrics['translation_rmse_m'],
            final_object_copy_rmse_m=final_metrics['object_copy_rmse_m'],
            online_trajectory_rmse_m=online_metrics['translation_rmse_m'],
            **batch_metrics,
            evaluated_future_poses=online_metrics['pose_count'],
            covariance_boundary_robots=sum(row['at_numerical_boundary'] for row in fits),
            update_s=float(t['update_s']), prediction_s=float(t['prediction_s']),
            calibration_s=float(t['calibration_s']),
            total_work_s=sum(float(t[key]) for key in ('update_s', 'prediction_s', 'calibration_s')),
            max_total_backlog_s=float(t['max_total_backlog_s'])))
    tables = [('summary', result), ('robots', robot_rows),
              ('innovation_diagnostics', noise_failure_diagnostics(predictions))]
    if args.native_residuals:
        tables.append(('failure_examples', independent_residual_examples(
            predictions, args.native_residuals, dataset, method=args.failure_method)))
    for name, rows in tables:
        with (args.output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        argv=__import__('sys').argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_manifest_sha256=hashlib.sha256((args.run / 'manifest.json').read_bytes()).hexdigest(),
        batch_manifest_sha256=(hashlib.sha256((args.batch_run / 'manifest.json').read_bytes()).hexdigest()
                               if args.batch_run else None),
        native_residual_hashes=({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in args.native_residuals.glob(f'd{dataset}_r*_raw.npz')}
                               if args.native_residuals else None),
        truth_used_by_estimator=False, evaluation_only=True,
        evaluation_interval='strictly after warmup; independently aligned per robot and arm',
        uncertainty='no iid significance claim across temporally correlated observations'), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
