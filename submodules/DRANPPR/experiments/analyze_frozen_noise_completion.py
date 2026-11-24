"""Posthoc summary and reference-consistency diagnosis; never estimator feedback."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.audit_conditional_replay import sha256
from experiments.audit_lazy_refresh import dump


def moments(values):
    if not values:
        return dict(count=0)
    x = np.asarray(values)
    return dict(count=len(values), mean=float(x.mean()), centered_sd=float(x.std()),
                rmse=float(np.sqrt(np.mean(x*x))), median=float(np.median(x)),
                absolute_p95=float(np.quantile(np.abs(x), .95)))


def reference_residuals(events, truth, labels, cutoff, stop):
    groups = {name:dict(bearing=[], range=[]) for name in ['prefix', 'suffix']}
    robots, unsupported = {}, 0
    for e in events:
        t = e['timestamp_s']
        if t > stop:
            break
        if e['kind'] != 'observation':
            continue
        r = e['robot']
        label = labels.get(e['landmark_subject'], e['landmark_subject'])
        if label not in truth.landmarks or not truth.robots[r].supports(t):
            unsupported += 1
            continue
        point, yaw = truth.robots[r].interpolate(t)
        delta = truth.landmarks[label]-point
        angle = e['bearing_rad']-(math.atan2(delta[1], delta[0])-yaw)
        errors = dict(bearing=math.atan2(math.sin(angle), math.cos(angle)),
                      range=e['range_m']-float(np.linalg.norm(delta)))
        group = 'prefix' if t <= cutoff else 'suffix'
        robot = robots.setdefault(str(r), {name:dict(bearing=[], range=[]) for name in groups})
        for key, value in errors.items():
            groups[group][key].append(value)
            robot[group][key].append(value)
    summarize = lambda data: {group:{key:moments(x) for key,x in values.items()} for group,values in data.items()}
    return dict(**summarize(groups), robots={r:summarize(data) for r,data in robots.items()}, unsupported=unsupported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    path = args.results
    manifest = json.loads((path/'manifest.json').read_text())
    if not manifest['all_estimators_terminal'] or any(r['exit_code'] != 0 for r in manifest['runs']):
        raise ValueError('unfinished estimator')
    analysis = json.loads((path/'analysis.json').read_text())
    warm = json.loads((path/'warm_start.json').read_text())
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    stop = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])+120.
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    root = Path(__file__).resolve().parents[1]
    data = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    source = audit_dataset(data, args.raw)
    truth = load_ground_truth(data, source.dataset_directory)
    labels = dict(data.evaluation.object_ground_truth_label_map)
    result = dict(cases={}, reference_errors={}, source_sha256=sha256(Path(__file__)),
        residual_scope='Posthoc reference errors include sensor, preprocessing, interpolation and reference error; not pure independent noise samples. No GT-based calibration or parameter selection.',
        comparisons_max={k:max(abs(r[k]) for r in analysis['comparisons'])
                         for k in ['cost_difference', 'max_xy_m', 'max_yaw_rad']})
    for case, online in analysis['online'].items():
        result['cases'][case] = {}
        for objective, item in online.items():
            f = next(r for r in analysis['completions'] if r['case'] == case and
                     r['objective'] == objective and r['start'] == objective and r['mode'] == 'central_lm')
            result['cases'][case][objective] = dict(
                online_causal_rmse_m=item['causal_quality']['translation_rmse_m'],
                common_alignment_causal_rmse_m=item['common_alignment_causal_rmse_m'],
                online_final_rmse_m=item['quality']['translation_rmse_m'],
                online_object_rmse_m=item['quality']['object_copy_rmse_m'],
                completed_final_rmse_m=f['quality']['translation_rmse_m'],
                completed_object_rmse_m=f['quality']['object_copy_rmse_m'],
                own_cost_reduction_fraction=1-f['summary']['final_cost']/f['summary']['initial_cost'])
        with Path(manifest['inputs'][case]['path']).open() as stream:
            result['reference_errors'][case] = reference_residuals(
                (json.loads(line) for line in stream), truth, labels, cutoff, stop)
    result['gaussian_mean_metrics'] = {
        objective:{metric:float(np.mean([rows[objective][metric] for case,rows in result['cases'].items()
                                        if case.startswith('gaussian')]))
                   for metric in result['cases']['original'][objective]}
        for objective in ['nominal', 'added']}
    dump(path/'diagnostics.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
