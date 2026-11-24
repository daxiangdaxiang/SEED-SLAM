"""Complete the identical arrived graph, without future observations or refitting."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_frozen_noise_completion import state_difference
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_support import apply_scales
from experiments.prepare_frozen_completion import prepare
from experiments.audit_online_calibration_baselines import benchmark_cases


def latest_pose_rows(rows, robots):
    selected, identities = {}, set()
    for row in rows:
        if row['kind'] != 'pose':
            continue
        robot = int(row['robot'])
        key = robot, int(row['index'])
        if key in identities or robot not in robots:
            raise ValueError('invalid pose support')
        identities.add(key)
        if robot not in selected or float(row['timestamp_s']) > float(selected[robot]['timestamp_s']):
            selected[robot] = row
    if set(selected) != set(robots):
        raise ValueError('missing robot pose')
    return [selected[r] for r in robots]


def checkpoint_times(manifest, warm, dataset, checkpoints=None):
    cases = dict(benchmark_cases(manifest))
    if dataset not in cases or warm['provenance']['dataset'] != dataset:
        raise ValueError('checkpoint dataset differs from source')
    first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])-first
    times = [30.,60.,90.] if checkpoints is None else list(map(float,checkpoints))
    if (not times or any(not np.isfinite(t) or not cutoff < t <= cases[dataset] for t in times)
            or any(a >= b for a,b in zip(times,times[1:]))):
        raise ValueError('invalid checkpoint times or insufficient suffix support')
    return times


def completion_geometry(before, after):
    from benchmarks.mrclam.materialize import estimate_se2_registration
    def keyed(rows):
        poses = [r for r in rows if r['kind']=='pose']
        result = {(int(r['robot']),int(r['index'])):[float(r['x']),float(r['y'])] for r in poses}
        if len(result) != len(poses):
            raise ValueError('duplicate pose support')
        return result
    a,b = keyed(before),keyed(after)
    if a.keys() != b.keys():
        raise ValueError('different pose support')
    left,right = np.array([a[k] for k in sorted(a)]),np.array([b[k] for k in sorted(a)])
    transform,deformation = estimate_se2_registration(left,right)
    raw = float(np.sqrt(np.mean(np.sum((right-left)**2,axis=1))))
    return dict(pose_count=len(a),raw_displacement_rms_m=raw,deformation_rms_m=deformation,
        rigid_explained_fraction=1.-deformation**2/raw**2 if raw else None,
        rotation_rad=transform.yaw,translation_m=[transform.x,transform.y],
        scope='Posthoc single rigid fit between estimates, no GT. Not an allowed gauge step with a fixed root and not applied to estimator outputs.')


def analyze_geometry(output):
    if not json.loads((output/'manifest.json').read_text())['all_terminal']:
        raise ValueError('unfinished checkpoint solves')
    records = []
    for row in json.loads((output/'producer.json').read_text()):
        before = read_rows(output/row['folder']/'final_estimates.csv')
        for mode in ('central_lm','central_gn'):
            after = read_rows(output/(row['folder']+'_'+mode)/'final_estimates.csv')
            records.append(dict(checkpoint=row['checkpoint'],arm=row['arm'],mode=mode,
                                **completion_geometry(before,after)))
    if (output/'geometry.json').exists():
        raise ValueError('geometry analysis already exists')
    write_json(output/'geometry.json',records)
    print(json.dumps(records,indent=2))


def run(source, output, dataset=1, checkpoints=None):
    source, output = source.resolve(), output.resolve()
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('source changed')
    source_manifest = json.loads((source/'manifest.json').read_text())
    warm = json.loads((source/f'd{dataset}/warm_start.json').read_text())
    checkpoints = checkpoint_times(source_manifest,warm,dataset,checkpoints)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest = dict(source=str(source),source_ledger_sha256=sha256(source/'SHA256SUMS'),
        dataset=dataset,checkpoints=checkpoints,runs=[],all_terminal=False,truth_used=False, binaries={},
        scope='Post-selected finite-work diagnostic. Complete exactly the arrived graph; not a real-time competitor or a new parameter fit.')
    for name in ('native-joint-replay','native-joint-audit'):
        path = root/'build/native/bin'/name
        manifest['binaries'][name] = dict(path=str(path),sha256=sha256(path))
        if name == 'native-joint-replay' and sha256(path) != source_manifest['binaries']['native']['sha256']:
            raise ValueError('checkpoint replay binary differs from source comparison')
        shutil.copy2(path,output/name)
    for name in ('experiments/audit_checkpoint_tracking.py','experiments/prepare_frozen_completion.py',
                 'experiments/network_noise_support.py','tests/test_checkpoint_tracking.py',
                 'docs/superpowers/plans/2026-11-17-tracking-and-se3-evidence.md'):
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    for name in ('warm_start.json','events.jsonl'):
        shutil.copy2(source/f'd{dataset}'/name,output/name)
    events = [json.loads(s) for s in (output/'events.jsonl').read_text().splitlines()]
    first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    scales = source_manifest['inputs'][str(dataset)]['scales']
    records = []

    def execute(command, tag):
        started = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_180s'
        manifest['runs'].append(dict(tag=tag,command=command,exit_code=code,seconds=time.perf_counter()-started))
        write_json(output/'manifest.json',manifest)
        if code:
            raise RuntimeError('checkpoint process failed; retained without retry')

    for start in manifest['checkpoints']:
        for arm in ('nominal','calibrated'):
            tag = f't{start:g}_{arm}'
            folder = output/tag
            command = [str(output/'native-joint-replay'),str(output/'events.jsonl'),str(folder),
                'joint_cached_one',str(start),'--warm-start',str(output/'warm_start.json'),'--lazy-refresh']
            q = scales if arm=='calibrated' else [1.]*5
            if arm=='calibrated':
                command += ['--suffix-variance-multipliers',*map(str,q)]
            command += [f'{a}:{b}' for a,b in warm['links']]
            execute(command,tag)
            prior = read_rows(source/f'd{dataset}'/arm/'causal_estimates.csv')
            expected = [r for r in prior if float(r['timestamp_s']) <= first+start]
            if read_rows(folder/'causal_estimates.csv') != expected:
                raise ValueError('truncated replay changed causal history')
            package = apply_scales(prepare(events,read_rows(folder/'final_estimates.csv'),warm,start,0.),q,cutoff,0.)
            write_json(output/(tag+'.json'),package)
            online = json.loads((folder/'summary.json').read_text())
            completed = {}
            for mode in ('central_lm','central_gn'):
                target = output/(tag+'_'+mode)
                execute([str(output/'native-joint-audit'),str(output/(tag+'.json')),str(target),mode,'100'],tag+'_'+mode)
                summary = json.loads((target/'summary.json').read_text())
                if abs(summary['initial_cost']-online['physical_cost'])/max(1.,online['physical_cost']) > 1e-10:
                    raise ValueError('frozen physical graph differs')
                completed[mode] = summary
            parity = state_difference(read_rows(output/(tag+'_central_lm')/'final_estimates.csv'),
                read_rows(output/(tag+'_central_gn')/'final_estimates.csv'))
            records.append(dict(checkpoint=start,arm=arm,folder=tag,online_summary=online,
                completion=completed,lm_gn_state_difference=parity,causal_prefix_exact=True))
            write_json(output/'producer.json',records)
            print(json.dumps(dict(checkpoint=start,arm=arm,cost=online['physical_cost'],
                completed_cost=completed['central_lm']['final_cost'],lm_gn=parity)),flush=True)
    manifest['all_terminal'] = True
    write_json(output/'manifest.json',manifest)


def evaluate(output, raw):
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import load_manifest,audit_dataset
    from experiments.evaluate_native_joint import evaluate_joint_rows,package_rows
    from experiments.analyze_online_calibration_baselines import position_errors
    manifest = json.loads((output/'manifest.json').read_text())
    if not manifest['all_terminal']:
        raise ValueError('unfinished checkpoint solves')
    root = Path(__file__).resolve().parents[1]
    data = load_manifest(root/f"benchmarks/mrclam/manifests/dataset{manifest.get('dataset',1)}.json")
    audited = audit_dataset(data,raw)
    truth = load_ground_truth(data,audited.dataset_directory)
    labels = dict(data.evaluation.object_ground_truth_label_map)
    warm = json.loads((output/'warm_start.json').read_text())
    robots = [r['id'] for r in warm['robots']]
    alignment = evaluate_joint_rows(package_rows(warm),truth,labels)
    records = []
    for producer in json.loads((output/'producer.json').read_text()):
        for mode in ('online','central_lm','central_gn'):
            tag = producer['folder']+('' if mode=='online' else '_'+mode)
            rows = read_rows(output/tag/'final_estimates.csv')
            quality = evaluate_joint_rows(rows,truth,labels)
            latest = latest_pose_rows(rows,robots)
            errors = position_errors(latest,truth,alignment)
            records.append(dict(checkpoint=producer['checkpoint'],arm=producer['arm'],mode=mode,
                quality=quality,latest_prefix_frame_rmse_m=float(np.sqrt(np.mean(errors**2))),
                latest_common_frame_errors_m=errors.tolist(),
                latest_whole_graph_aligned_rmse_m=float(np.sqrt(np.mean(position_errors(latest,truth,quality)**2)))))
            print(json.dumps(records[-1]),flush=True)
    write_json(output/'comparison.json',dict(records=records,common_prefix_alignment=alignment,
        truth_used_only_by_evaluation=True,scope='Five latest poses per checkpoint, not a whole-trajectory causal metric. Completed methods use NO future observations.'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','evaluate','geometry'])
    parser.add_argument('--source',type=Path,default=Path('results/dranppr_online_calibration_baselines_20261029'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--dataset',type=int,default=1)
    parser.add_argument('--checkpoints',type=float,nargs='+')
    args=parser.parse_args()
    if args.mode=='run':
        run(args.source,args.output,args.dataset,args.checkpoints)
    elif args.mode=='geometry':
        analyze_geometry(args.output)
    else:
        if args.raw is None:
            parser.error('--raw required for evaluation')
        evaluate(args.output,args.raw)
