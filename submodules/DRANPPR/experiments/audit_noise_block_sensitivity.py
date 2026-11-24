"""Fixed-chart block-deletion noise sensitivity, never a deployable rejection rule."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.audit_native_checkpoint import sha256
from experiments.audit_lazy_refresh import dump
from experiments.audit_network_noise_transfer import prediction_summary
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_support import assemble, components, apply_scales


def deletion_groups(data, package):
    if data.get('truth_used', False) or package.get('truth_used', False):
        raise ValueError('truth-assisted source is forbidden')
    if data['schema'] != 'dranppr_physical_linearization_v1':
        raise ValueError('invalid physical linearization')
    times, labels, expected = {}, {}, Counter()
    roots = {}
    for robot in package['robots']:
        rid = robot['id']
        roots[rid] = min(p['index'] for p in robot['poses'])
        for p in robot['poses']:
            times[rid,p['index']] = p['timestamp_s']
        for obj in robot['objects']:
            labels[rid,obj['cluster']] = obj['label']
        expected.update((rid,f['pose'],f['cluster']) for f in robot['observations'])
    records = []
    for index,factor in enumerate(data['factors']):
        if factor['kind'] != 'observation':
            continue
        poses = [t['index'] for t in factor['terms'] if t['kind']=='x']
        objects = [t['index'] for t in factor['terms'] if t['kind']=='l']
        rid = factor['robot']
        if len(objects)!=1 or len(poses)>1 or (not poses and rid!=package['root_robot']):
            raise ValueError('unmatched physical observation terms')
        pose = poses[0] if poses else roots[rid]
        key = (rid,pose,objects[0])
        if expected[key] <= 0:
            raise ValueError('physical observations do not match package')
        expected[key] -= 1
        records.append(dict(index=index, robot=rid, pose=pose,
            time=times[rid,pose], label=labels[rid,objects[0]], cluster=objects[0]))
    if any(expected.values()) or not records:
        raise ValueError('package observations do not match physical export')
    frames = sorted({(r['time'],r['robot'],r['pose']) for r in records})
    if len(frames) < 3:
        raise ValueError('three complete frame blocks require at least three frames')
    def group(tag, family, selected, **extra):
        return dict(tag=tag, family=family, removed=[r['index'] for r in selected],
            removed_observations=len(selected),
            removed_frames=len({(r['robot'],r['pose']) for r in selected}),
            robots=sorted({r['robot'] for r in selected}),
            clusters=sorted({r['cluster'] for r in selected}),**extra)
    output = [group('full','full',[])]
    for label in sorted({r['label'] for r in records}):
        output.append(group(f'subject_{label}','subject',[r for r in records if r['label']==label],label=label))
    for block,indices in enumerate(np.array_split(np.arange(len(frames)),3)):
        chosen = {frames[i] for i in indices}
        selected = [r for r in records if (r['time'],r['robot'],r['pose']) in chosen]
        output.append(group(f'frames_{block}','frames',selected,
            first_time=min(r['time'] for r in selected),last_time=max(r['time'] for r in selected)))
    return output


def retained_model(data, removed):
    if (len(set(removed)) != len(removed) or
            any(not isinstance(i,int) or i<0 or i>=len(data['factors']) for i in removed)):
        raise ValueError('invalid deleted factor indices')
    if any(data['factors'][i]['kind']!='observation' for i in removed):
        raise ValueError('only observation deletion is allowed')
    removed = set(removed)
    return dict(data,factors=[f for i,f in enumerate(data['factors']) if i not in removed])


def fit_model(data):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from dranppr_variance_components import ContrastVarianceModel
    a,b,groups = assemble(data)
    d,base = components(groups)
    model = ContrastVarianceModel(a,b,d,base)
    result = model.fit(np.zeros(5))
    rank = len(a)-len(model.z)
    result.update(scales=result['components'],rows=a.shape[0],columns=a.shape[1],rank=rank,
        nullity=a.shape[1]-rank,initial_nll=model.nll(np.zeros(5)),
        truth_used=False,fixed_chart=True)
    return result


def compare_predictions(candidate, reference):
    def keys(rows):
        return [(r['robot'],r['source_row'],r['subject'],r['timestamp_s']) for r in rows]
    k = keys(candidate)
    if not k or len(set(k))!=len(k) or k!=keys(reference):
        raise ValueError('forecast support differs')
    one,two = prediction_summary(candidate),prediction_summary(reference)
    result = dict(candidate=one,reference=two,delta_mean_nll=one['mean_nll']-two['mean_nll'])
    result['by_robot'] = []
    for robot in sorted({r['robot'] for r in candidate}):
        a = prediction_summary([r for r in candidate if r['robot']==robot])
        b = prediction_summary([r for r in reference if r['robot']==robot])
        result['by_robot'].append(dict(robot=robot,**a,delta_mean_nll=a['mean_nll']-b['mean_nll']))
    return result


def fit_worker(source, output, dataset, tag):
    linear_path = source/f'd{dataset}/training/linearization.json'
    data = json.loads(linear_path.read_text())
    package = json.loads((source/f'd{dataset}/warm_start.json').read_text())
    group = next(g for g in deletion_groups(data,package) if g['tag']==tag)
    result = fit_model(retained_model(data,group['removed']))
    result.update(group=group,source_sha256=sha256(linear_path))
    folder = output/f'd{dataset}/{tag}'
    folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'fit.json',result)
    if not result['success']:
        raise RuntimeError('calibration failed; no fit retry')


def predict_worker(source, output, dataset, tag, elapsed):
    from experiments.network_noise_prediction import forecast
    case = output/f'd{dataset}/{tag}'
    fitted = json.loads((case/'fit.json').read_text())
    warm = json.loads((source/f'd{dataset}/warm_start.json').read_text())
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
    original = source/f'd{dataset}/t{elapsed}_nominal.json'
    package = apply_scales(json.loads(original.read_text()),fitted['scales'],cutoff,0.)
    path = case/f't{elapsed}.json'
    dump(path,package)
    folder = case/f't{elapsed}'
    command = [str(output/'native-joint-audit'),str(path),str(folder),'central_lm','100','--linearization']
    start = time.perf_counter()
    process = subprocess.run(command,check=True,timeout=120,capture_output=True,text=True)
    (case/f't{elapsed}_solver.log').write_text(process.stdout+process.stderr)
    solve_seconds = time.perf_counter()-start
    events = [json.loads(s) for s in (source/f'd{dataset}/events.jsonl').read_text().splitlines()]
    predicted = forecast(package,read_rows(folder/'final_estimates.csv'),
        json.loads((folder/'linearization.json').read_text()),events,first+elapsed,5.,fitted['scales'])
    reference = json.loads((source/f'd{dataset}/t{elapsed}_joint_qr/forecast.json').read_text())
    comparison = compare_predictions(predicted['predictions'],reference['predictions'])
    predicted.update(dataset=dataset,tag=tag,elapsed=elapsed,solve_seconds=solve_seconds,
        worker_seconds=time.perf_counter()-start,original_package_sha256=sha256(original),
        comparison=comparison)
    if tag=='full':
        comparison['max_mean_difference'] = max(float(np.max(np.abs(np.array(a['mean'])-b['mean'])))
            for a,b in zip(predicted['predictions'],reference['predictions']))
        comparison['max_covariance_difference'] = max(float(np.max(np.abs(np.array(a['covariance'])-b['covariance'])))
            for a,b in zip(predicted['predictions'],reference['predictions']))
        if max(comparison['max_mean_difference'],comparison['max_covariance_difference'])>1e-6:
            raise ValueError('full-fit forecast parity failed')
    dump(folder/'forecast.json',predicted)


def run(source, output):
    source,output = source.resolve(),output.resolve()
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if sha256(source/name)!=digest:
            raise ValueError('source checksum mismatch: '+name)
        hashes[name] = digest
    output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    names = ['experiments/audit_noise_block_sensitivity.py','experiments/network_noise_prediction.py',
        'experiments/network_noise_support.py','experiments/audit_network_noise_transfer.py',
        'scripts/dranppr_variance_components.py','scripts/dranppr_linear_evidence.py',
        'tests/test_noise_block_sensitivity.py','docs/superpowers/plans/2026-10-04-noise-block-sensitivity.md']
    for name in names:
        target=output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    shutil.copy2(source/'native-joint-audit',output/'native-joint-audit')
    plan = {}
    for dataset in (1,4):
        plan[str(dataset)] = deletion_groups(json.loads((source/f'd{dataset}/training/linearization.json').read_text()),
            json.loads((source/f'd{dataset}/warm_start.json').read_text()))
    dump(output/'deletion_plan.json',plan)
    manifest = dict(source=str(source),source_hashes=hashes,runs=[],all_workers_terminal=False,
        truth_used=False,model_changed=False,future_selected=False,
        binary_sha256=sha256(output/'native-joint-audit'),
        libraries=subprocess.check_output(['ldd',str(output/'native-joint-audit')],text=True),
        intended_fits=sum(map(len,plan.values())),timeout_s=180,
        scope='Fixed-chart calibration sensitivity, complete physical graph for checkpoint solves, no GT or winner selection')
    dump(output/'manifest.json',manifest)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def execute(mode,dataset,tag,elapsed=None):
        command=[sys.executable,'-m','experiments.audit_noise_block_sensitivity',mode,
            '--source',str(source),'--output',str(output),'--dataset',str(dataset),'--tag',tag]
        label=f'd{dataset}_{tag}_{mode}'
        if elapsed is not None:
            command += ['--elapsed',str(elapsed)]
            label += f'_{elapsed}'
        started=time.perf_counter()
        with (output/(label+'.log')).open('x') as log:
            try:
                code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:
                code='timeout_180s'
        row=dict(mode=mode,dataset=dataset,tag=tag,elapsed=elapsed,command=command,
            exit_code=code,wall_seconds=time.perf_counter()-started)
        manifest['runs'].append(row)
        dump(output/'manifest.json',manifest)
        print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
        return code==0
    successful=[]
    for dataset in (1,4):
        for group in plan[str(dataset)]:
            tag=group['tag']
            if execute('fit',dataset,tag):
                fit=json.loads((output/f'd{dataset}/{tag}/fit.json').read_text())
                if tag=='full':
                    prior=json.loads((source/f'd{dataset}/fits/joint_qr/fit.json').read_text())
                    if np.max(np.abs(np.log(np.array(fit['scales'])/prior['scales'])))>1e-6:
                        raise ValueError('full-fit parameter parity failed')
                successful.append((dataset,tag))
            elif tag=='full':
                raise RuntimeError('full-fit control failed')
    results=[]
    for dataset,tag in successful:
        records=[]
        for elapsed in (30,60,90):
            if execute('predict',dataset,tag,elapsed):
                records.extend(json.loads((output/f'd{dataset}/{tag}/t{elapsed}/forecast.json').read_text())['predictions'])
            else:
                break
        else:
            reference=[]
            for elapsed in (30,60,90):
                reference.extend(json.loads((source/f'd{dataset}/t{elapsed}_joint_qr/forecast.json').read_text())['predictions'])
            result=compare_predictions(records,reference)
            fit=json.loads((output/f'd{dataset}/{tag}/fit.json').read_text())
            full=json.loads((output/f'd{dataset}/full/fit.json').read_text())
            result.update(dataset=dataset,tag=tag,fit=fit,
                scale_ratio=(np.array(fit['scales'])/full['scales']).tolist())
            results.append(result)
            dump(output/'analysis.json',dict(results=results,truth_used=False))
    manifest.update(all_workers_terminal=True,successful_fits=len(successful),complete_forecast_arms=len(results))
    dump(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','fit','predict'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=int,choices=[1,4])
    parser.add_argument('--tag')
    parser.add_argument('--elapsed',type=int,choices=[30,60,90])
    args=parser.parse_args()
    if args.mode=='run':
        run(args.source,args.output)
    elif args.dataset is None or args.tag is None:
        parser.error('worker requires dataset and tag')
    elif args.mode=='fit':
        fit_worker(args.source,args.output,args.dataset,args.tag)
    elif args.elapsed is None:
        parser.error('prediction worker requires elapsed')
    else:
        predict_worker(args.source,args.output,args.dataset,args.tag,args.elapsed)


if __name__=='__main__':
    main()
