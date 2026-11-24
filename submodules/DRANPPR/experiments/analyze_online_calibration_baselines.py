"""Posthoc parity, physical-weight and causal-error checks; never tune an estimator."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest
from experiments.audit_mean_evidence import sha256, write_json
from experiments.evaluate_incremental_reference import read_rows
from experiments.evaluate_native_joint import evaluate_joint_rows, package_rows
from experiments.network_noise_support import apply_scales
from experiments.prepare_frozen_completion import prepare
from experiments.audit_online_calibration_baselines import benchmark_cases


def position_errors(rows, truth, alignment):
    angle = alignment['alignment_yaw']
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c,-s],[s,c]])
    translation = np.array([alignment['alignment_x'],alignment['alignment_y']])
    return np.array([np.linalg.norm(rotation@[float(r['x']),float(r['y'])]+translation-
        truth.robots[int(r['robot'])].interpolate(float(r['timestamp_s']))[0]) for r in rows])


def analyze(output, raw):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((output/'manifest.json').read_text())
    comparison = json.loads((output/'comparison.json').read_text())
    if not manifest['all_estimators_terminal']:
        raise ValueError('unfinished producers')
    verified, parity, groups, audits = [], [], [], []
    for name, entry in manifest['binaries'].items():
        if sha256(entry['path']) != entry['sha256'] or sha256(output/(name+'.binary')) != entry['sha256']:
            raise ValueError('changed experiment binary')
        for path, digest in entry['library_hashes'].items():
            if sha256(path) != digest:
                raise ValueError('changed runtime library')
    cases = benchmark_cases(manifest)
    historical = []
    for dataset, _ in cases:
        if dataset in (2,3):
            historical.append((dataset,'nominal',root/f'results/dranppr_lazy_transfer_stress_20261003/d{dataset}_clean/candidate1'))
        elif dataset in (4,9):
            historical.append((dataset,'nominal',root/f'results/dranppr_full_lazy_refresh_20261003/d{dataset}/candidate1'))
            for arm in ('imesa_paper_one','rimesa_one'):
                if any(r['dataset']==dataset and r['arm']==arm for r in comparison['records']):
                    historical.append((dataset,arm,root/f'results/dranppr_native_online_20260923/d{dataset}_one'/arm))
    for dataset, arm, old in historical:
        for name in ('final_estimates.csv','causal_estimates.csv'):
            same = sha256(output/f'd{dataset}'/arm/name) == sha256(old/name)
            parity.append(dict(dataset=dataset, arm=arm, file=name, archived=str(old/name), identical=same))
            if not same:
                raise ValueError('state replay differs from archive')
    for dataset, duration in cases:
        case = output/f'd{dataset}'
        warm = json.loads((case/'warm_start.json').read_text())
        first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
        events = [json.loads(s) for s in (case/'events.jsonl').read_text().splitlines()]
        data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audit = audit_dataset(data, raw)
        truth = load_ground_truth(data, audit.dataset_directory)
        labels = dict(data.evaluation.object_ground_truth_label_map)
        prefix_alignment = evaluate_joint_rows(package_rows(warm), truth, labels)
        for record in comparison['records']:
            if record['dataset'] != dataset:
                continue
            arm = record['arm']
            folder = case/arm
            causal = read_rows(folder/'causal_estimates.csv')
            final = read_rows(folder/'final_estimates.csv')
            keys = [(r['robot'],r['index'],r['timestamp_s']) for r in causal]
            if len(set(keys)) != len(keys):
                raise ValueError('duplicate causal state')
            index = {(r['robot'],r['index'],r['timestamp_s']):r for r in final if r['kind']=='pose'}
            matched_final = [index[k] for k in keys]
            updates = read_rows(folder/'updates.csv')
            if [(int(r['robot']),float(r['timestamp_s'])) for r in causal] != [
                    (int(r['robot']),float(r['timestamp_s'])) for r in updates]:
                raise ValueError('causal publication/frame mismatch')
            error = position_errors(causal, truth, record['causal_quality'])
            prefix_error = position_errors(causal, truth, prefix_alignment)
            final_frame_error = position_errors(causal, truth, record['quality'])
            revised_error = position_errors(matched_final, truth, record['quality'])
            revisions = np.array([np.hypot(float(a['x'])-float(b['x']),float(a['y'])-float(b['y']))
                                  for a,b in zip(causal, matched_final)])
            elapsed = np.array([float(r['timestamp_s'])-first for r in causal])
            robots = np.array([int(r['robot']) for r in causal])
            selectors = [('all', np.ones(len(causal), dtype=bool))]
            selectors += [(f'robot{r}',robots==r) for r in range(1,6)]
            selectors += [(f't{t}_{min(t+30,duration):g}',(elapsed>t)&(elapsed<=t+30))
                          for t in range(0,math.ceil(duration),30)]
            for group, select in selectors:
                if not select.any():
                    continue
                groups.append(dict(dataset=dataset,arm=arm,group=group,count=int(select.sum()),
                    causal_rmse_m=float(np.sqrt(np.mean(error[select]**2))),
                    prefix_aligned_causal_rmse_m=float(np.sqrt(np.mean(prefix_error[select]**2))),
                    terminal_aligned_causal_rmse_m=float(np.sqrt(np.mean(final_frame_error[select]**2))),
                    terminal_same_support_rmse_m=float(np.sqrt(np.mean(revised_error[select]**2))),
                    causal_max_error_m=float(error[select].max()),
                    max_terminal_revision_m=float(revisions[select].max())))
            if arm not in ('nominal','calibrated'):
                continue
            package = prepare(events, final, warm, duration, 0.)
            scales = manifest['inputs'][str(dataset)]['scales'] if arm=='calibrated' else [1.]*5
            package = apply_scales(package, scales, cutoff, 0.)
            path = case/(arm+'_terminal_audit.json')
            write_json(path, package)
            target = case/(arm+'_terminal_audit')
            command = [str(root/'build/native/bin/native-joint-audit'),str(path),str(target),'central_lm','1']
            with (case/(arm+'_terminal_audit.log')).open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            audits.append(dict(command=command, exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError('terminal physical audit failed')
            initial = json.loads((target/'summary.json').read_text())['initial_cost']
            expected = record['summary']['physical_cost']
            discrepancy = abs(initial-expected)/max(1.,abs(expected))
            verified.append(dict(dataset=dataset,arm=arm,initial_audit_cost=initial,
                                 online_physical_cost=expected,relative_error=discrepancy))
            if discrepancy > 1e-10:
                raise ValueError('online physical weights differ from frozen graph')
    result = dict(parity=parity, terminal_physical_checks=verified, posthoc_groups=groups,
        commands=audits, truth_used_only_for_analysis=True,
        scope='Post-selected descriptions; group errors reuse one all-robot global transform, no per-group realignment. Terminal audit solves excluded from comparison budgets; only initial costs checked.')
    write_json(output/'verification.json', result)
    for row in groups:
        if row['arm'] in ('nominal','calibrated'):
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.output.resolve(), args.raw)
