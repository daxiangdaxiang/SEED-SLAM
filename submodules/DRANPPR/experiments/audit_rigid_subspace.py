"""Paired same-objective online replay of the opt-in rigid interface subspace."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_native_online import verify
from experiments.audit_online_calibration_baselines import benchmark_cases, common_credits, evaluate
from experiments.evaluate_incremental_reference import read_rows


def budget_audit(baseline, candidate):
    credits = common_credits(baseline, candidate)
    left = right = exceedances = maximum = 0
    for a, b in zip(baseline, candidate):
        left += int(a['optimizer_bytes'])+int(a['control_bytes'])
        right += int(b['optimizer_bytes'])+int(b['control_bytes'])
        exceedances += right > left
        maximum = max(maximum, right-left)
    return dict(frames=len(credits), baseline_final_bytes=left, candidate_final_bytes=right,
                prefix_exceedances=exceedances, max_prefix_excess_bytes=maximum)


def run(source, output, datasets, candidate='rigid'):
    if candidate not in ('rigid','fused'):
        raise ValueError('invalid candidate')
    source, output = source.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[1]
    original = json.loads((source/'manifest.json').read_text())
    cases = benchmark_cases(original, datasets)
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(source/name) != digest:
            raise ValueError('source archive changed: '+name)
    output.mkdir(parents=True, exist_ok=False)
    binary = root/'build/native/bin/native-joint-replay'
    shutil.copy2(binary, output/'native.binary')
    names = ['experiments/audit_rigid_subspace.py', 'experiments/online_stream.h',
             'experiments/native_local/joint_step.h', 'experiments/native_local/joint_replay.cpp',
             'experiments/native_local/joint_package.h', 'cmake/native_local_model.cmake',
             'tests/testNativeInterface.cpp', 'tests/test_native_rigid_subspace.py']
    for directory in ('src','include/DPGO'):
        names += [str(p.relative_to(root)) for p in (root/directory).glob('Native*')]
        names += [str(p.relative_to(root)) for p in (root/directory).glob('Incremental*')]
    for name in names:
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    manifest = dict(source=str(source), runs=[], benchmark_cases=[dict(dataset=d,duration=t) for d,t in cases],
        binary_sha256=sha256(binary), all_estimators_terminal=False, truth_used_by_estimator=False,
        ldd=subprocess.check_output(['ldd',str(binary)],text=True), timeout_s=600,
        protocol='Same frozen weights/events/root and one solve per frame. Actual every-hop bytes, no equal-time claim.',
        inputs={}, budget_audits={}, historical_baseline_reproduction={},
        candidate=candidate, candidate_state_parity={})
    write_json(output/'manifest.json',manifest)
    records = []
    for dataset, duration in cases:
        case = output/f'd{dataset}'
        case.mkdir()
        for name in ('events.jsonl','warm_start.json'):
            shutil.copy2(source/f'd{dataset}'/name,case/name)
        package = json.loads((case/'warm_start.json').read_text())
        if package.get('truth_used',False) or package['provenance']['event_sha256'] != sha256(case/'events.jsonl'):
            raise ValueError('invalid causal input provenance')
        scales = original['inputs'][str(dataset)]['scales']
        manifest['inputs'][str(dataset)] = dict(scales=scales,
            events_sha256=sha256(case/'events.jsonl'),package_sha256=sha256(case/'warm_start.json'))
        for weights in ('nominal','calibrated'):
            if weights=='calibrated' and scales is None:
                continue
            for solver in ('pcg',candidate):
                arm = weights+'_'+solver
                command = [str(output/'native.binary'),str(case/'events.jsonl'),str(case/arm),
                    'joint_cached_one',str(duration),'--warm-start',str(case/'warm_start.json'),'--lazy-refresh']
                command += [f'{a}:{b}' for a,b in package['links']]
                if weights=='calibrated':
                    command += ['--suffix-variance-multipliers',*map(str,scales)]
                if solver=='rigid':
                    command += ['--rigid-subspace']
                if solver=='fused':
                    command += ['--fused-one-step']
                start = time.perf_counter()
                with (output/f'd{dataset}_{arm}.log').open('x') as log:
                    try:
                        code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
                    except subprocess.TimeoutExpired:
                        code = 'timeout_600s'
                manifest['runs'].append(dict(tag=f'd{dataset}_{arm}',command=command,exit_code=code,
                                            seconds=time.perf_counter()-start))
                write_json(output/'manifest.json',manifest)
                print(json.dumps(manifest['runs'][-1]),flush=True)
                if code != 0:
                    raise RuntimeError('replay failed; diagnostics retained')
            left, right = [read_rows(case/(weights+'_'+s)/'updates.csv') for s in ('pcg',candidate)]
            credits = common_credits(left,right)
            reference = json.loads((case/(weights+'_pcg')/'summary.json').read_text())
            manifest['budget_audits'][f'd{dataset}_{weights}'] = budget_audit(left,right)
            for solver in ('pcg',candidate):
                arm = weights+'_'+solver
                summary = verify(case/arm,package,reference,credits)
                if summary['consensus_max_m']>1e-10:
                    raise ValueError('shared copies disagree')
                records.append(dict(dataset=dataset,arm=arm,folder=f'd{dataset}/{arm}',summary=summary))
            manifest['historical_baseline_reproduction'][f'd{dataset}_{weights}'] = {
                name: sha256(case/(weights+'_pcg')/name)==sha256(source/f'd{dataset}'/weights/name)
                for name in ('final_estimates.csv','causal_estimates.csv')}
            if candidate=='fused':
                parity = {name: sha256(case/(weights+'_pcg')/name)==sha256(case/(weights+'_fused')/name)
                          for name in ('final_estimates.csv','causal_estimates.csv')}
                manifest['candidate_state_parity'][f'd{dataset}_{weights}'] = parity
                if not all(parity.values()):
                    raise ValueError('fused one-step changed published estimates')
            write_json(output/'producer.json',records)
            write_json(output/'manifest.json',manifest)
    manifest['all_estimators_terminal'] = True
    write_json(output/'manifest.json',manifest)


def fixed_frame_audit(output, raw):
    import numpy as np
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.analyze_online_calibration_baselines import position_errors
    from experiments.evaluate_native_joint import evaluate_joint_rows, package_rows
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((output/'manifest.json').read_text())
    comparison = json.loads((output/'comparison.json').read_text())
    if not manifest['all_estimators_terminal']:
        raise ValueError('unfinished producers')
    records = []
    for dataset, duration in benchmark_cases(manifest):
        data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        truth = load_ground_truth(data,audit_dataset(data,raw).dataset_directory)
        warm = json.loads((output/f'd{dataset}/warm_start.json').read_text())
        alignment = evaluate_joint_rows(package_rows(warm),truth,dict(data.evaluation.object_ground_truth_label_map))
        for row in comparison['records']:
            if row['dataset']!=dataset:
                continue
            causal = read_rows(output/row['folder']/'causal_estimates.csv')
            supported = [r for r in causal if truth.robots[int(r['robot'])].supports(float(r['timestamp_s']))]
            errors = position_errors(supported,truth,alignment)
            records.append(dict(dataset=dataset,arm=row['arm'],pose_count=len(supported),
                unsupported=len(causal)-len(supported),rmse_m=float(np.sqrt(np.mean(errors**2))),
                max_m=float(errors.max())))
    write_json(output/'fixed_frame.json',dict(records=records,
        protocol='One common registration from the identical warm prefix, never refitted to suffix or per arm. Posthoc GT only.'))
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','evaluate','fixed_frame'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--datasets',type=int,nargs='+')
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--candidate',choices=['rigid','fused'],default='rigid')
    args = parser.parse_args()
    if args.mode=='run':
        run(args.source,args.output,args.datasets,args.candidate)
    elif args.mode=='evaluate':
        evaluate(args.output,args.raw)
    else:
        fixed_frame_audit(args.output,args.raw)
