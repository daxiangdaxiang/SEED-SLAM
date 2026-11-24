"""Posthoc common-prefix-alignment sensitivity check, not estimator feedback."""
import argparse
import json
import math
from pathlib import Path

from experiments.audit_conditional_replay import read_rows, sha256
from experiments.audit_lazy_refresh import dump
from experiments.analyze_full_lazy_refresh import temporal_profile


def point_error_squared(row, target, alignment):
    x, y = float(row['x']), float(row['y'])
    angle, tx, ty = (alignment[k] for k in ['alignment_yaw', 'alignment_x', 'alignment_y'])
    if not all(math.isfinite(v) for v in [x, y, angle, tx, ty, *target]):
        raise ValueError('nonfinite point or alignment')
    c, s = math.cos(angle), math.sin(angle)
    return (c*x-s*y+tx-target[0])**2+(s*x+c*y+ty-target[1])**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results/'manifest.json').read_text())
    if not manifest['all_estimators_terminal'] or any(r['exit_code'] != 0 for r in manifest['runs']):
        raise ValueError('unfinished or failed estimator run')
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows, package_rows
    root = Path(__file__).resolve().parents[1]
    data_manifest = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    source = audit_dataset(data_manifest, args.raw)
    truth = load_ground_truth(data_manifest, source.dataset_directory)
    warm = json.loads((args.results/'warm_start.json').read_text())
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    alignment = evaluate_joint_rows(package_rows(warm), truth,
                                   dict(data_manifest.evaluation.object_ground_truth_label_map))
    analysis = json.loads((args.results/'analysis.json').read_text())
    output = dict(common_prefix_alignment=alignment, cases={}, source_sha256=sha256(Path(__file__)),
        scope='Posthoc sensitivity analysis: one GT alignment from the identical common past, frozen for all suffixes. No suffix state used to fit alignment; no GT supplied to estimator. Equal-count quarters, no re-alignment.')
    for case, runs in analysis['results'].items():
        output['cases'][case] = {}
        for tag in runs:
            folder = args.results/case/tag
            causal, updates = read_rows(folder/'causal_estimates.csv'), read_rows(folder/'updates.csv')
            identity = lambda row: (row['robot'], row['timestamp_s'])
            if list(map(identity, causal)) != list(map(identity, updates)):
                raise ValueError('causal/service identity mismatch')
            errors = []
            for row in causal:
                series, t = truth.robots[int(row['robot'])], float(row['timestamp_s'])
                errors.append(point_error_squared(row, series.interpolate(t)[0], alignment)
                              if series.supports(t) else None)
            supported = [e for e in errors if e is not None]
            output['cases'][case][tag] = dict(
                common_alignment_causal_rmse_m=math.sqrt(sum(supported)/len(supported)),
                quarters=temporal_profile(updates, errors))
            final_groups = {name: [] for name in ['prefix', 'suffix']}
            final_alignment = runs[tag]['quality']
            for row in read_rows(folder/'final_estimates.csv'):
                if row['kind'] != 'pose':
                    continue
                t, series = float(row['timestamp_s']), truth.robots[int(row['robot'])]
                if series.supports(t):
                    group = 'prefix' if t <= cutoff else 'suffix'
                    final_groups[group].append(point_error_squared(row, series.interpolate(t)[0], final_alignment))
            combined = [e for group in final_groups.values() for e in group]
            if not math.isclose(math.sqrt(sum(combined)/len(combined)),
                                final_alignment['translation_rmse_m'], rel_tol=1e-10):
                raise ValueError('final error decomposition mismatch')
            output['cases'][case][tag]['final_prefix_suffix'] = {
                name:dict(count=len(values), rmse_m=math.sqrt(sum(values)/len(values)))
                for name,values in final_groups.items()}
            output['cases'][case][tag]['final_prefix_suffix_alignment'] = (
                'One final whole-trajectory alignment per run, not separately fitted to prefix/suffix')
    dump(args.results/'common_alignment.json', output)
    print(json.dumps({case:{tag:r['common_alignment_causal_rmse_m'] for tag,r in runs.items()}
                      for case,runs in output['cases'].items()}))


if __name__ == '__main__':
    main()
