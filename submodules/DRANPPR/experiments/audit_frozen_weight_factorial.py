"""Actual fixed-weight factorial solves; no fitting, tuning or GT evaluation."""
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

from experiments.analyze_motion_time_structure import split_score
from experiments.audit_frozen_noise_completion import state_difference
from experiments.audit_lazy_refresh import dump
from experiments.audit_motion_time_structure import verify_archive
from experiments.audit_native_checkpoint import sha256
from experiments.audit_network_noise_transfer import prediction_summary
from experiments.audit_range_prediction_attribution import matched_predictions
from experiments.evaluate_incremental_reference import read_rows


def weight_grid(scales):
    scales=np.asarray(scales,float)
    if scales.shape!=(5,) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('invalid frozen scales')
    return {f'q{q}_b{b}_r{r}':(scales[:3].tolist() if q else [1.]*3)+
            [float(scales[3]) if b else 1.,float(scales[4]) if r else 1.]
            for q in (0,1) for b in (0,1) for r in (0,1)}


def endpoint_parity(forecast, reference):
    mean=covariance=nll=0.
    for a,b in matched_predictions(forecast,reference):
        d=np.asarray(a['mean'])-b['mean'];d[0]=math.atan2(math.sin(d[0]),math.cos(d[0]))
        mean=max(mean,float(np.max(np.abs(d))))
        covariance=max(covariance,float(np.max(np.abs(np.asarray(a['covariance'])-b['covariance']))))
        nll=max(nll,abs(a['nll']-b['nll']))
    if max(mean,covariance,nll)>1e-6:
        raise ValueError('frozen endpoint parity failed')
    return dict(max_mean=mean,max_covariance=covariance,max_nll=nll)


def run(source, output):
    started=time.perf_counter();source=source.resolve();output=output.resolve()
    hashes=verify_archive(source)
    old=json.loads((source/'manifest.json').read_text())
    for path,digest in old['library_hashes'].items():
        if sha256(Path(path))!=digest:
            raise ValueError('runtime library changed: '+path)
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'native-joint-audit';shutil.copy2(source/binary.name,binary)
    root=Path(__file__).resolve().parents[1]
    for name in ('experiments/audit_frozen_weight_factorial.py','experiments/audit_noise_block_sensitivity.py',
                 'experiments/network_noise_prediction.py','experiments/network_noise_support.py',
                 'experiments/audit_frozen_noise_completion.py','tests/test_frozen_weight_factorial.py',
                 'docs/superpowers/plans/2026-10-05-range-prediction-attribution.md'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    manifest=dict(source=str(source),source_hashes=hashes,library_hashes=old['library_hashes'],
        binary_sha256=sha256(binary),runs=[],complete=False,truth_used=False,parameters_fitted=False,
        intended_predictions=48,intended_gn_controls=48,iteration_cap=100,worker_timeout_s=180,
        producer_budget_s=900,scope='Frozen group toggles; actual centralized solves, not online or tuned selection.')
    dump(output/'manifest.json',manifest)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def execute(command, log, **tags):
        remaining=900-(time.perf_counter()-started)
        if remaining<=0:
            raise RuntimeError('producer budget exhausted')
        begin=time.perf_counter()
        with log.open('x') as stream:
            try:
                code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=min(180.,remaining)).returncode
            except subprocess.TimeoutExpired:
                code='timeout'
        record=dict(command=command,exit_code=code,wall_seconds=time.perf_counter()-begin,**tags)
        manifest['runs'].append(record);dump(output/'manifest.json',manifest)
        if code!=0:
            raise RuntimeError('worker failed: '+str(log))
        return record
    results=[];controls=[];endpoints=[];solvers=[]
    for dataset in (1,4):
        fitted=json.loads((source/f'd{dataset}/fits/joint_qr/fit.json').read_text())
        if fitted['truth_used'] or not fitted['success']:
            raise ValueError('invalid existing calibration')
        for tag,scales in weight_grid(fitted['scales']).items():
            case=output/f'd{dataset}/{tag}';case.mkdir(parents=True)
            dump(case/'fit.json',dict(scales=scales,parameters_fitted=False,source_fit=f'd{dataset}/fits/joint_qr/fit.json'))
            predictions=[]
            for elapsed in (30,60,90):
                command=[sys.executable,'-m','experiments.audit_noise_block_sensitivity','predict',
                    '--source',str(source),'--output',str(output),'--dataset',str(dataset),'--tag',tag,'--elapsed',str(elapsed)]
                execute(command,case/f't{elapsed}_worker.log',dataset=dataset,tag=tag,elapsed=elapsed,mode='LM_predict')
                folder=case/f't{elapsed}';forecast=json.loads((folder/'forecast.json').read_text())
                predictions.extend(forecast['predictions'])
                summary=json.loads((folder/'summary.json').read_text())
                solvers.append(dict(dataset=dataset,tag=tag,elapsed=elapsed,
                    gradient=summary['posthoc_gradient_norm'],lm_iterations=summary['lm_iterations'],
                    covariance_solve_relative_residual=forecast['covariance_solve_relative_residual']))
                if tag in ('q0_b0_r0','q1_b1_r1'):
                    arm='nominal' if tag=='q0_b0_r0' else 'joint_qr'
                    reference=json.loads((source/f'd{dataset}/t{elapsed}_{arm}/forecast.json').read_text())
                    endpoints.append(dict(dataset=dataset,tag=tag,elapsed=elapsed,**endpoint_parity(forecast,reference)))
                else:
                    reference=json.loads((source/f'd{dataset}/t{elapsed}_nominal/forecast.json').read_text())
                    matched_predictions(forecast,reference)
                gn=case/f't{elapsed}_gn'
                command=[str(binary),str(folder.with_suffix('.json')),str(gn),'central_gn','100']
                execute(command,case/f't{elapsed}_gn.log',dataset=dataset,tag=tag,elapsed=elapsed,mode='GN_control')
                gn_summary=json.loads((gn/'summary.json').read_text())
                controls.append(dict(dataset=dataset,tag=tag,elapsed=elapsed,
                    **state_difference(read_rows(folder/'final_estimates.csv'),read_rows(gn/'final_estimates.csv')),
                    cost_difference=gn_summary['final_cost']-summary['final_cost']))
            for robot in (0,1,2,3,4,5):
                rows=[p for p in predictions if robot==0 or p['robot']==robot]
                scores=[split_score(p['innovation'],p['covariance']) for p in rows]
                results.append(dict(dataset=dataset,tag=tag,robot=robot,scales=scales,
                    **prediction_summary(rows),
                    range_nll=float(np.mean([p['range'] for p in scores])),
                    bearing_nll=float(np.mean([p['bearing'] for p in scores])),
                    range_rmse=float(np.sqrt(np.mean([p['innovation'][1]**2 for p in rows]))),
                    range_variance=float(np.mean([p['covariance'][1][1] for p in rows]))))
            dump(output/'analysis.json',dict(results=results,controls=controls,endpoints=endpoints,solvers=solvers))
            print(json.dumps(dict(dataset=dataset,tag=tag,predictions=len(predictions),
                wall_seconds=time.perf_counter()-started)),flush=True)
    manifest.update(complete=True,wall_seconds=time.perf_counter()-started)
    dump(output/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
