"""Sparse REML parity/scalability and genuine SE3 synthetic transfer controls."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import qr

from experiments.audit_mean_evidence import sha256, write_json
from experiments.network_noise_support import assemble, components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from dranppr_sparse_variance_components import SparseVarianceModel
from dranppr_variance_components import ContrastVarianceModel


def execute(command, output, manifest, tag):
    begun = time.perf_counter()
    with (output/(tag+'.log')).open('x') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout_600s'
    record = dict(tag=tag, command=command, exit_code=code, seconds=time.perf_counter()-begun)
    manifest['runs'].append(record)
    write_json(output/'manifest.json', manifest)
    print(json.dumps(record), flush=True)
    return code == 0


def snapshot(output):
    for name in ('experiments/audit_sparse_calibration.py','experiments/audit_network_noise_support.py',
                 'experiments/network_noise_support.py','scripts/dranppr_sparse_variance_components.py',
                 'scripts/dranppr_variance_components.py','scripts/dranppr_linear_evidence.py',
                 'experiments/requirements-sparse-calibration.txt',
                 'experiments/native_local/pose3_calibration.cpp','cmake/native_local_model.cmake',
                 'tests/test_sparse_variance_components.py','tests/test_network_noise_support.py',
                 'tests/test_pose3_calibration.py','docs/superpowers/plans/2026-11-17-sparse-calibration-3d.md'):
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    import sparseqr
    import sparseqr._sparseqr as extension
    libraries = subprocess.check_output(['ldd',extension.__file__],text=True)
    write_json(output/'dependencies.json',dict(sparseqr_version=importlib.metadata.version('sparseqr'),
        numpy_version=np.__version__,scipy_version=importlib.metadata.version('scipy'),
        extension_sha256=sha256(extension.__file__),libraries=libraries,
        package_files={str(p.relative_to(Path(sparseqr.__file__).parent)):sha256(p)
            for p in Path(sparseqr.__file__).parent.glob('*') if p.is_file()}))


def seal(output):
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))


def qr_check(source, fit_path, output):
    a, y, labels = assemble(json.loads(source.read_text()))
    d, base = components(labels)
    fitted = json.loads(fit_path.read_text())
    q = np.asarray(fitted['scales'])
    # Independent square-root reference, no normal equations or error contrasts.
    a /= np.linalg.norm(a, axis=0)
    _, r = qr(a, mode='economic', check_finite=False)
    reference = 2*np.log(np.abs(np.diag(r))).sum()
    variance = base+d@q
    weighted = a/np.sqrt(variance[:, None])
    orthogonal, triangular = qr(weighted, mode='economic', check_finite=False)
    z = y/np.sqrt(variance)
    residual = z-orthogonal@(orthogonal.T@z)
    nll = .5*(np.log(variance).sum()+2*np.log(np.abs(np.diag(triangular))).sum()-reference+
              residual@residual+(len(y)-a.shape[1])*np.log(2*np.pi))
    gradient = .5*q*(d.T@((1.-np.sum(orthogonal**2,axis=1)-residual**2)/variance))
    value = dict(nll=float(nll), score=gradient.tolist(),
        nll_abs_error=float(abs(nll-fitted['nll'])),
        score_max_error=float(np.max(np.abs(gradient-fitted['log_score']))),
        scope='Independent weighted QR at fitted scales, no covariance inverse or normal equations')
    write_json(output, value)
    if value['nll_abs_error'] > 1e-6 or value['score_max_error'] > 1e-6:
        raise ValueError('sparse/QR likelihood or score mismatch')


def run_real(source, output):
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('source archive changed')
    previous_manifest = json.loads((source/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    manifest = dict(source=str(source), source_ledger_sha256=sha256(source/'SHA256SUMS'),
        benchmark_cases=[dict(dataset=2,duration=120),dict(dataset=3,duration=120),dict(dataset=9,duration=300)],
        runs=[], all_training_terminal=False, truth_used=False,
        scope='Same archived physical Jacobians, variance family, bounds and optimizer; only REML linear algebra changes')
    parity = []
    for case in manifest['benchmark_cases']:
        dataset = case['dataset']
        current, old = output/f'd{dataset}', source/f'd{dataset}'
        current.mkdir()
        for name in ('warm_start.json','events.jsonl'):
            shutil.copy2(old/name,current/name)
        linearization = old/'training/linearization.json'
        if not execute([sys.executable,'-m','experiments.audit_network_noise_support','fit',
            '--source',str(linearization),'--arm','joint_qr','--implementation','sparse',
            '--output',str(current/'fits/joint_qr')],output,manifest,f'd{dataset}_fit'):
            continue
        new = json.loads((current/'fits/joint_qr/fit.json').read_text())
        reference = old/'fits/joint_qr/fit.json'
        if reference.exists():
            previous = json.loads(reference.read_text())
            previous_status = next(r['exit_code'] for r in previous_manifest['runs']
                                   if r['tag'] == f'd{dataset}_fit')
            difference = dict(dataset=dataset,reference_process_exit_code=previous_status,
                reference_partial_output=previous_status != 0,
                nll_abs_error=abs(new['nll']-previous['nll']),
                log_scale_max_error=float(np.max(np.abs(np.log(np.array(new['scales'])/previous['scales'])))),
                fisher_relative_error=float(np.linalg.norm(np.array(new['fisher_eigenvalues'])-previous['fisher_eigenvalues'])/
                                            np.linalg.norm(previous['fisher_eigenvalues'])))
            parity.append(difference)
            if difference['nll_abs_error'] > 1e-6 or difference['log_scale_max_error'] > 1e-4:
                raise ValueError('original contrast fit mismatch')
        if not execute([sys.executable,'-m','experiments.audit_sparse_calibration','qr_check',
            '--source',str(linearization),'--fit',str(current/'fits/joint_qr/fit.json'),
            '--output',str(current/'qr_check.json')],output,manifest,f'd{dataset}_qr_check'):
            raise ValueError('independent square-root validation failed')
    manifest['all_training_terminal'] = True
    write_json(output/'manifest.json',manifest)
    write_json(output/'parity.json',parity)
    seal(output)


def run_pose3(output):
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    binary = ROOT/'build/native/bin/native-pose3-calibration-audit'
    shutil.copy2(binary,output/'native-pose3-calibration-audit')
    manifest = dict(runs=[], seeds=list(range(31,41)), regimes=['nominal','heterogeneous'],
        binary_sha256=sha256(binary), all_terminal=False,
        scope='Synthetic full SE3 pose/object batch graph; training/test independent measurement noise, same geometry and common perturbed-truth starts. No online or real-data claim.')
    records = []
    for regime in manifest['regimes']:
        for seed in manifest['seeds']:
            case = output/f'{regime}_{seed}'
            if not execute([str(binary),str(case/'nominal'),str(seed),regime],output,manifest,f'{regime}_{seed}_nominal'):
                raise RuntimeError('3D nominal failed')
            data = json.loads((case/'nominal/training.json').read_text())
            a,y = np.asarray(data['a']),np.asarray(data['b'])
            d = np.eye(4)[data['groups']]
            model = SparseVarianceModel(a,y,d,np.zeros(len(y)))
            started = time.perf_counter()
            fit = model.fit(np.zeros(4))
            fit['fit_seconds'] = time.perf_counter()-started
            reference = ContrastVarianceModel(a,y,d,np.zeros(len(y)))
            theta = np.log(fit['components'])
            parity = dict(nll_abs_error=abs(reference.nll(theta)-fit['nll']),
                score_max_error=float(np.max(np.abs(reference.gradient(theta)-fit['log_score']))))
            if not fit['success'] or parity['nll_abs_error'] > 1e-6 or parity['score_max_error'] > 1e-6:
                raise ValueError('3D fit/reference mismatch')
            write_json(case/'fit.json',dict(fit=fit,parity=parity))
            nominal = json.loads((case/'nominal/summary.json').read_text())
            row = dict(regime=regime,seed=seed,fit=fit,parity=parity,nominal=nominal)
            for arm,scales in [('calibrated',fit['components']),('oracle',nominal['generating_scales'])]:
                if not execute([str(binary),str(case/arm),str(seed),regime,*map(str,scales)],output,manifest,f'{regime}_{seed}_{arm}'):
                    raise RuntimeError('3D arm failed')
                if sha256(case/arm/'training.json') != sha256(case/'nominal/training.json'):
                    raise ValueError('3D noise intervention changed training input')
                if sha256(case/arm/'test_input.json') != sha256(case/'nominal/test_input.json'):
                    raise ValueError('3D noise intervention changed test observations or initialization')
                row[arm] = json.loads((case/arm/'summary.json').read_text())
            records.append(row)
            write_json(output/'comparison.json',records)
    manifest['all_terminal'] = True
    write_json(output/'manifest.json',manifest)
    seal(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['real','pose3','qr_check'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--fit',type=Path)
    args = parser.parse_args()
    if args.mode == 'real':
        run_real(args.source.resolve(),args.output.resolve())
    elif args.mode == 'pose3':
        run_pose3(args.output.resolve())
    else:
        qr_check(args.source,args.fit,args.output)
