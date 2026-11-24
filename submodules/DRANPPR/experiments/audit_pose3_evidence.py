"""Exact-linear noise-evidence controls on an archived physical SE3 Jacobian."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import sparse
from scipy.linalg import lstsq, qr

from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_sparse_calibration import execute, seal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dranppr_sparse_variance_components import SparseVarianceModel

SEEDS = list(range(401,441))
SCALES = {'nominal':np.ones(4),'heterogeneous':np.array([.1,4.,9.,.25])}


def sample_control(a, labels, scales, seed, repetitions):
    a = sparse.csr_matrix(a,dtype=float)
    labels,scales = np.asarray(labels),np.asarray(scales,float)
    if (type(repetitions) is not int or repetitions < 1 or labels.shape != (a.shape[0],) or
            not np.issubdtype(labels.dtype,np.integer) or scales.ndim != 1 or
            np.any(labels < 0) or np.any(labels >= len(scales)) or
            not np.isfinite(scales).all() or np.any(scales <= 0.)):
        raise ValueError('invalid independent repetition design')
    training = np.random.default_rng(np.random.SeedSequence([seed,0]))
    testing = np.random.default_rng(np.random.SeedSequence([seed,1]))
    repeated = np.tile(labels,repetitions)
    return dict(a=sparse.vstack([a]*repetitions,format='csr'),labels=repeated,
        training_y=training.normal(size=len(repeated))*np.sqrt(scales[repeated]),
        test_y=testing.normal(size=len(labels))*np.sqrt(scales[labels]))


def prediction_risk(a, y, used_variance, true_variance):
    a,y = np.asarray(a,float),np.asarray(y,float)
    variance,true = np.asarray(used_variance,float),np.asarray(true_variance,float)
    if (a.ndim != 2 or y.shape != (a.shape[0],) or variance.shape != y.shape or true.shape != y.shape or
            any(not np.isfinite(x).all() for x in (a,y,variance,true)) or
            np.any(variance <= 0.) or np.any(true <= 0.)):
        raise ValueError('invalid risk model')
    state,_,rank,_ = lstsq(a/np.sqrt(variance[:,None]),y/np.sqrt(variance),lapack_driver='gelsy')
    if rank != a.shape[1]:
        raise ValueError('unidentified test state')
    # The generated state is zero; this is mean prediction error, not residual loss.
    return float(np.mean((a@state)**2/true))


def dense_qr_score(a, y, d, q):
    variance = d@q
    _,r0 = qr(a,mode='economic')
    orthogonal,r = qr(a/np.sqrt(variance[:,None]),mode='economic')
    z = y/np.sqrt(variance)
    residual = z-orthogonal@(orthogonal.T@z)
    nll = .5*(np.log(variance).sum()+2*np.log(np.abs(np.diag(r))).sum()-
        2*np.log(np.abs(np.diag(r0))).sum()+residual@residual+(len(y)-a.shape[1])*np.log(2*np.pi))
    score = .5*q*(d.T@((1.-np.sum(orthogonal**2,axis=1)-residual**2)/variance))
    return float(nll),score


def verify(source, output):
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('changed evidence archive')
    data = json.loads((source/'physical_design.json').read_text())
    a,labels = np.asarray(data['a']),np.asarray(data['groups'],dtype=int)
    records = json.loads((source/'records.json').read_text())
    checks = []
    for row in records:
        if row['failure'] is not None:
            continue
        sample = sample_control(a,labels,SCALES[row['regime']],row['seed'],row['repetitions'])
        d = np.eye(4)[sample['labels']]
        value,score = dense_qr_score(sample['a'].toarray(),sample['training_y'],d,np.asarray(row['fit']['components']))
        difference = dict(regime=row['regime'],seed=row['seed'],repetitions=row['repetitions'],
            nll_abs_error=abs(value-row['fit']['nll']),
            score_max_error=float(np.max(np.abs(score-row['fit']['log_score']))))
        if difference['nll_abs_error'] > 1e-6 or difference['score_max_error'] > 1e-6:
            raise ValueError('independent dense QR mismatch')
        checks.append(difference)
    result = dict(source_ledger_sha256=sha256(source/'SHA256SUMS'),checks=checks,
        attempted=len(records),checked=len(checks),
        max_nll_abs_error=max(r['nll_abs_error'] for r in checks),
        max_score_error=max(r['score_max_error'] for r in checks))
    if output.exists():
        raise ValueError('verification output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


def worker(source, output, regime, repetitions):
    output.mkdir(parents=True,exist_ok=False)
    data = json.loads(source.read_text())
    a,labels = np.asarray(data['a']),np.asarray(data['groups'],dtype=int)
    q = SCALES[regime]
    initial = sample_control(a,labels,q,SEEDS[0],repetitions)
    d = np.eye(4)[initial['labels']]
    model = SparseVarianceModel(initial['a'],initial['training_y'],d,np.zeros(len(d)))
    information = model.fisher(np.log(q))
    write_json(output/'design_information.json',dict(
        rows=len(d),columns=a.shape[1],residual_dof=model.residual_dof,
        fisher_at_generating_scales=information.tolist(),
        local_log_scale_standard_errors=np.sqrt(np.diag(np.linalg.inv(information))).tolist(),
        note='Oracle design information, not fitted confidence or a boundary-valid interval.'))
    records = []
    for seed in SEEDS:
        sample = sample_control(a,labels,q,seed,repetitions)
        row = dict(seed=seed,regime=regime,repetitions=repetitions,generating_scales=q.tolist(),
            nominal_risk=prediction_risk(a,sample['test_y'],np.ones(len(labels)),q[labels]),
            oracle_risk=prediction_risk(a,sample['test_y'],q[labels],q[labels]),
            calibrated_risk=None,fit=None,failure=None)
        started = time.perf_counter()
        try:
            model = SparseVarianceModel(sample['a'],sample['training_y'],d,np.zeros(len(d)))
            fit = model.fit(np.zeros(4))
            row['fit'] = fit
            if fit['success']:
                row['calibrated_risk'] = prediction_risk(a,sample['test_y'],
                    np.asarray(fit['components'])[labels],q[labels])
            else:
                row['failure'] = fit['message']
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            row['failure'] = type(error).__name__+': '+str(error)
        row['fit_seconds'] = time.perf_counter()-started
        records.append(row)
        write_json(output/'records.json',records)
        print(json.dumps(dict(seed=seed,regime=regime,repetitions=repetitions,
            failure=row['failure'],boundary=row['fit']['boundary_components'] if row['fit'] else None)),flush=True)


def run(source, output):
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('changed physical SE3 archive')
    output.mkdir(parents=True,exist_ok=False)
    design = source/'heterogeneous_35/nominal/training.json'
    shutil.copy2(design,output/'physical_design.json')
    for name in ('experiments/audit_pose3_evidence.py','experiments/audit_sparse_calibration.py',
                 'scripts/dranppr_sparse_variance_components.py','tests/test_pose3_evidence.py',
                 'docs/superpowers/plans/2026-11-17-tracking-and-se3-evidence.md'):
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    shutil.copy2(source/'dependencies.json',output/'dependencies_from_producer.json')
    manifest = dict(source=str(source),source_ledger_sha256=sha256(source/'SHA256SUMS'),
        design_sha256=sha256(design),seeds=SEEDS,repetitions=[1,4],regimes=list(SCALES),runs=[],
        all_terminal=False,all_declared_fits=False,
        scope='Post-selected physical SE3 fixed-design exact-linear Gaussian control. Generating variances known only to simulation and scoring. No nonlinear, online, real-data or outlier claim.',
        test_support='Original 1x design in both arms. Same independent test draw per seed/regime; only training evidence grows.')
    records = []
    for regime in SCALES:
        for repetitions in manifest['repetitions']:
            tag = f'{regime}_{repetitions}'
            success = execute([sys.executable,'-m','experiments.audit_pose3_evidence','worker',
                '--source',str(output/'physical_design.json'),'--output',str(output/tag),
                '--regime',regime,'--repetitions',str(repetitions)],output,manifest,tag)
            if (output/tag/'records.json').exists():
                records.extend(json.loads((output/tag/'records.json').read_text()))
            if not success:
                print('Retained incomplete worker: '+tag,flush=True)
    manifest['all_terminal'] = True
    manifest['all_declared_fits'] = len(records) == len(SEEDS)*len(SCALES)*len(manifest['repetitions'])
    write_json(output/'manifest.json',manifest)
    write_json(output/'records.json',records)
    summary = []
    for regime in SCALES:
        for repetitions in manifest['repetitions']:
            rows = [r for r in records if r['regime']==regime and r['repetitions']==repetitions]
            valid = [r for r in rows if r['failure'] is None]
            if not valid:
                summary.append(dict(regime=regime,repetitions=repetitions,attempts=len(rows),successes=0))
                continue
            estimate = np.array([r['fit']['components'] for r in valid])
            summary.append(dict(regime=regime,repetitions=repetitions,attempts=len(rows),successes=len(valid),
                boundary_seeds=[r['seed'] for r in valid if r['fit']['at_numerical_boundary']],
                mean_scales=estimate.mean(axis=0).tolist(),median_scales=np.median(estimate,axis=0).tolist(),
                relative_scale_rmse=np.sqrt(np.mean((estimate/SCALES[regime]-1.)**2,axis=0)).tolist(),
                paired_mean_risk={arm:float(np.mean([r[arm+'_risk'] for r in valid]))
                    for arm in ('nominal','calibrated','oracle')},
                calibrated_risk_wins=sum(r['calibrated_risk'] < r['nominal_risk'] for r in valid),
                note='Risk statistics use only declared successful pairs; attempts/failures are reported separately.'))
    write_json(output/'summary.json',summary)
    seal(output)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','worker','verify'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regime',choices=list(SCALES))
    parser.add_argument('--repetitions',type=int)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.source.resolve(),args.output.resolve())
    elif args.mode == 'verify':
        verify(args.source.resolve(),args.output.resolve())
    else:
        worker(args.source.resolve(),args.output.resolve(),args.regime,args.repetitions)
