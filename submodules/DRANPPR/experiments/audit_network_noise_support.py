"""Past-only network covariance attribution; not an online adaptive backend."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.audit_conditional_replay import read_rows, sha256
from experiments.audit_lazy_refresh import dump
from experiments.audit_frozen_noise_completion import state_difference
from experiments.network_noise_support import assemble, components, apply_scales


def fit(source, arm, output, implementation='dense'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from dranppr_variance_components import ContrastVarianceModel
    if implementation == 'sparse':
        from dranppr_sparse_variance_components import SparseVarianceModel
    data = json.loads(source.read_text())
    a, b, groups = assemble(data, split=arm == 'split_qr', sparse_output=implementation == 'sparse')
    observation_only = arm == 'joint_r'
    d, base = components(groups, observation_only)
    model = (SparseVarianceModel if implementation == 'sparse' else ContrastVarianceModel)(a, b, d, base)
    result = model.fit(np.zeros(d.shape[1]))
    q = result['components']
    result.update(arm=arm, scales=([1.,1.,1.]+q if observation_only else q),
        initial_nll=model.nll(np.zeros(d.shape[1])), rows=a.shape[0], columns=a.shape[1],
        rank=a.shape[0]-result['residual_dof'], source_sha256=sha256(source), truth_used=False,
        fixed_linearization=True, noise_parameter_groups_shared_across_robots=True,
        warning='Classical REML, finite numerical log bounds [-16,10], no prior. Boundary estimates are not a precision certificate. Split/joint NLLs refer to different contrasts.')
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'fit.json', result)
    if implementation == 'dense':
        np.savez_compressed(output/'contrast.npz', z=model.z, base=model.base, components=model.components)
    print(json.dumps(result), flush=True)
    if not result['success']:
        raise RuntimeError('variance optimizer did not report success')


def run(output, raw):
    root = Path(__file__).resolve().parents[1]
    prior = root/'results/dranppr_frozen_noise_completion_20261003'
    previous = json.loads((prior/'manifest.json').read_text())
    old_analysis = json.loads((prior/'analysis.json').read_text())
    if not previous['all_estimators_terminal']:
        raise ValueError('incomplete source experiment')
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'native-joint-audit'
    shutil.copy2('/tmp/dranppr_backend_build/bin/native-joint-audit', binary)
    shutil.copy2(prior/'warm_start.json', output/'warm_start.json')
    if sha256(output/'warm_start.json') != previous['warm_start_sha256']:
        raise ValueError('prefix hash mismatch')
    warm = json.loads((output/'warm_start.json').read_text())
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    names = ['experiments/audit_network_noise_support.py', 'experiments/network_noise_support.py',
             'experiments/audit_frozen_noise_completion.py', 'experiments/evaluate_native_joint.py',
             'scripts/dranppr_variance_components.py', 'scripts/dranppr_linear_evidence.py',
             'tests/test_network_noise_support.py', 'CMakeLists.txt', 'cmake/native_local_model.cmake',
             'docs/superpowers/plans/2026-10-03-network-noise-support.md']
    sources = [root/name for name in names]+list((root/'experiments/native_local').glob('*'))
    for directory, suffix in [('src','.cpp'),('include/DPGO','.h')]:
        for stem in ['Native*','Incremental*']:
            sources += list((root/directory).glob(stem+suffix))
    for source in sources:
        if source.is_file():
            target = output/'source'/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(runs=[], inputs={}, source_archive=str(prior), source_manifest_sha256=sha256(prior/'manifest.json'),
        binary_sha256=sha256(binary), warm_start_sha256=sha256(output/'warm_start.json'),
        libraries=subprocess.check_output(['ldd', str(binary)], text=True),
        timeout_s=600, max_iterations=100, all_estimators_terminal=False, truth_used_by_estimator=False,
        calibration='Single common prefix, fixed-linear REML, no future measurement or GT input',
        warning='D2 development diagnostic; frozen endpoint accuracy, not causal online performance. Injected variance is supplied externally. Calibration is centralized, not a distributed implementation.')
    dump(output/'manifest.json', manifest)

    def execute(command, tag):
        start = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_600s'
        record = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(record)
        dump(output/'manifest.json', manifest)
        print(json.dumps(record), flush=True)
        if code != 0:
            raise RuntimeError('run failed; evidence retained; no retuning/retry')

    execute([str(binary),str(output/'warm_start.json'),str(output/'training'),'central_lm','100','--linearization'], 'training')
    source = output/'training/linearization.json'
    linear = json.loads(source.read_text())
    a, b, _ = assemble(linear)
    training = json.loads((output/'training/summary.json').read_text())
    if not math.isclose(.5*float(b@b), training['final_cost'], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('native physical loss/export mismatch')
    if not math.isclose(float(np.linalg.norm(a.T@b)),training['posthoc_gradient_norm'],rel_tol=1e-8,abs_tol=1e-8):
        raise ValueError('native gradient/export mismatch')
    fitted = {'nominal':dict(scales=[1.]*5, truth_used=False)}
    for arm in ['joint_r','joint_qr','split_qr']:
        execute([sys.executable,'-m','experiments.audit_network_noise_support','fit',
                 '--source',str(source),'--arm',arm,'--output',str(output/'fits'/arm)], 'fit_'+arm)
        fitted[arm] = json.loads((output/'fits'/arm/'fit.json').read_text())
    runs, parity = [], []
    for case in ['original','gaussian_20261003','gaussian_20261004','gaussian_20261005']:
        # Every arm starts from the SAME saved nominal online endpoint.
        tag = case+'_nominal_from_nominal_central_lm'
        original = prior/(tag+'.json')
        expected = next(r['package_sha256'] for r in old_analysis['completions'] if r['tag']==tag)
        if sha256(original) != expected:
            raise ValueError('source frozen package hash mismatch')
        package = json.loads(original.read_text())
        input_path = output/(case+'_original_input.json')
        shutil.copy2(original, input_path)
        manifest['inputs'][case] = dict(source=str(original), sha256=expected)
        added = .0225 if case.startswith('gaussian') else 0.
        for arm in ['nominal','joint_r','joint_qr','split_qr']:
            weighted = apply_scales(package, fitted[arm]['scales'], cutoff, added)
            package_path = output/(case+'_'+arm+'.json')
            dump(package_path, weighted)
            modes = ['central_lm','central_gn'] if case in ['original','gaussian_20261003'] else ['central_lm']
            for mode in modes:
                name = '_'.join([case,arm,mode])
                folder = output/'solves'/name
                execute([str(binary),str(package_path),str(folder),mode,'100'], name)
                summary = json.loads((folder/'summary.json').read_text())
                runs.append(dict(case=case,arm=arm,mode=mode,folder=str(folder),summary=summary,
                                 package_sha256=sha256(package_path)))
                if arm == 'nominal' and mode == 'central_lm':
                    old_objective = 'added' if added else 'nominal'
                    reference = next(r for r in old_analysis['completions'] if r['case']==case and
                                     r['objective']==old_objective and r['start']=='nominal' and r['mode']=='central_lm')
                    difference = state_difference(read_rows(folder/'final_estimates.csv'),
                                                   read_rows(Path(reference['folder'])/'final_estimates.csv'))
                    loss_difference = abs(summary['final_cost']-reference['summary']['final_cost'])
                    if difference['max_xy_m']>1e-7 or loss_difference>1e-8:
                        raise ValueError('nominal completion changed relative to archived control')
                    parity.append(dict(case=case,**difference,cost_abs=loss_difference))
    manifest['all_estimators_terminal'] = True
    dump(output/'manifest.json', manifest)
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    data = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    dataset = audit_dataset(data, raw)
    truth = load_ground_truth(data, dataset.dataset_directory)
    labels = dict(data.evaluation.object_ground_truth_label_map)
    comparisons = []
    for result in runs:
        rows = read_rows(Path(result['folder'])/'final_estimates.csv')
        result['quality'] = evaluate_joint_rows(rows, truth, labels)
        if result['mode']=='central_gn':
            ref = next(r for r in runs if r['case']==result['case'] and r['arm']==result['arm'] and r['mode']=='central_lm')
            comparisons.append(dict(case=result['case'],arm=result['arm'],
                cost_difference=result['summary']['final_cost']-ref['summary']['final_cost'],
                **state_difference(rows,read_rows(Path(ref['folder'])/'final_estimates.csv'))))
    dump(output/'analysis.json',dict(fits=fitted,runs=runs,nominal_parity=parity,solver_comparisons=comparisons,
        truth_used_by_evaluation=True,warning=manifest['warning']))
    print(json.dumps(dict(solves=len(runs),output=str(output/'analysis.json'))),flush=True)


def intervene(source, output, raw):
    producer = json.loads((source/'manifest.json').read_text())
    analysis = json.loads((source/'analysis.json').read_text())
    if not producer['all_estimators_terminal']:
        raise ValueError('incomplete primary experiment')
    output.mkdir(parents=True,exist_ok=False)
    binary = output/'native-joint-audit'
    shutil.copy2(source/'native-joint-audit',binary)
    if sha256(binary) != producer['binary_sha256']:
        raise ValueError('primary binary hash mismatch')
    warm = json.loads((source/'warm_start.json').read_text())
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    manifest = dict(source=str(source),primary_manifest_sha256=sha256(source/'manifest.json'),
        primary_analysis_sha256=sha256(source/'analysis.json'),binary_sha256=sha256(binary),
        post_selected=True,refitted=False,truth_used_by_estimator=False,all_estimators_terminal=False,
        scope='Q-only interventions on already fitted joint/split QR; R restored to nominal plus known injection variance',runs=[])
    results = []
    for case in ['original','gaussian_20261003','gaussian_20261004','gaussian_20261005']:
        package = json.loads((source/(case+'_original_input.json')).read_text())
        for origin in ['joint_qr','split_qr']:
            scales = analysis['fits'][origin]['scales'][:3]+[1.,1.]
            package_path = output/(case+'_'+origin+'_q_only.json')
            dump(package_path,apply_scales(package,scales,cutoff,.0225 if case.startswith('gaussian') else 0.))
            for mode in (['central_lm','central_gn'] if case in ['original','gaussian_20261003'] else ['central_lm']):
                tag = case+'_'+origin+'_q_only_'+mode
                folder = output/tag
                command = [str(binary),str(package_path),str(folder),mode,'100']
                start = time.perf_counter()
                with (output/(tag+'.log')).open('x') as log:
                    try:
                        code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
                    except subprocess.TimeoutExpired:
                        code = 'timeout_600s'
                record = dict(tag=tag,command=command,exit_code=code,wall_seconds=time.perf_counter()-start)
                manifest['runs'].append(record)
                dump(output/'manifest.json',manifest)
                print(json.dumps(record),flush=True)
                if code != 0:
                    raise RuntimeError('intervention failed; retained without retuning')
                results.append(dict(case=case,origin=origin,mode=mode,folder=str(folder),scales=scales,
                    summary=json.loads((folder/'summary.json').read_text()),package_sha256=sha256(package_path)))
    manifest['all_estimators_terminal'] = True
    dump(output/'manifest.json',manifest)
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset,load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    root = Path(__file__).resolve().parents[1]
    data = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    dataset = audit_dataset(data,raw)
    truth = load_ground_truth(data,dataset.dataset_directory)
    labels = dict(data.evaluation.object_ground_truth_label_map)
    for result in results:
        result['quality'] = evaluate_joint_rows(read_rows(Path(result['folder'])/'final_estimates.csv'),truth,labels)
    dump(output/'analysis.json',dict(runs=results,truth_used_by_evaluation=True,post_selected=True))
    print(json.dumps(dict(solves=len(results),output=str(output/'analysis.json'))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    fitting = commands.add_parser('fit')
    fitting.add_argument('--source',type=Path,required=True)
    fitting.add_argument('--arm',choices=['joint_r','joint_qr','split_qr'],required=True)
    fitting.add_argument('--output',type=Path,required=True)
    fitting.add_argument('--implementation',choices=['dense','sparse'],default='dense')
    running = commands.add_parser('run')
    running.add_argument('--output',type=Path,required=True)
    running.add_argument('--raw',type=Path,required=True)
    intervention = commands.add_parser('intervene')
    intervention.add_argument('--source',type=Path,required=True)
    intervention.add_argument('--output',type=Path,required=True)
    intervention.add_argument('--raw',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='fit':
        fit(args.source,args.arm,args.output.resolve(),args.implementation)
    elif args.command=='intervene':
        intervene(args.source.resolve(),args.output.resolve(),args.raw)
    else:
        run(args.output.resolve(),args.raw)


if __name__=='__main__':
    main()
