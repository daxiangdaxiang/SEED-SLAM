"""Bounded sampling-uncertainty sensitivity, separate from online comparisons."""
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
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from experiments.analyze_network_noise_transfer import rescore
from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_rolling_prediction import identity
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_prediction import forecast
from experiments.network_noise_support import apply_scales


def parameter_covariance(base, components, scales):
    c = base + np.einsum('k,kij->ij', scales, components)
    factor = cho_factor(c, lower=True)
    derivative = np.array([q*cho_solve(factor, v) for q, v in zip(scales, components)])
    information = .5*np.einsum('kij,lji->kl', derivative, derivative)
    covariance = cho_solve(cho_factor(information, lower=True), np.eye(len(scales)))
    return covariance, information


def cubature(mean, covariance):
    directions = math.sqrt(len(mean))*np.linalg.cholesky(covariance).T
    return np.concatenate((mean+directions, mean-directions))


def mix_predictions(components):
    if not components or not components[0]:
        raise ValueError('empty predictive mixture')
    for rows in components:
        rescore(rows, components[0])
    output = []
    for rows in zip(*components):
        means = np.array([r['mean'] for r in rows])
        mean = means.mean(axis=0)
        mean[0] = math.atan2(np.sin(means[:, 0]).mean(), np.cos(means[:, 0]).mean())
        delta = means-mean
        delta[:, 0] = np.arctan2(np.sin(delta[:, 0]), np.cos(delta[:, 0]))
        covariance = np.mean([r['covariance'] for r in rows], axis=0)+delta.T@delta/len(rows)
        innovation = np.array(rows[0]['mean'])+rows[0]['innovation']-mean
        innovation[0] = math.atan2(math.sin(innovation[0]), math.cos(innovation[0]))
        output.append(dict(zip(('robot','source_row','timestamp_s','subject'), identity(rows[0])),
            mean=mean.tolist(), innovation=innovation.tolist(), covariance=covariance.tolist(),
            nll=float(-logsumexp([-r['nll'] for r in rows])+math.log(len(rows)))))
    return output


def statistics(rows):
    errors = np.array([r['innovation'] for r in rows])
    return dict(count=len(rows), nll=float(np.mean([r['nll'] for r in rows])),
        range_rmse_m=float(np.sqrt(np.mean(errors[:, 1]**2))),
        range_signed_error_m=float(errors[:, 1].mean()),
        mean_range_sd_m=float(np.mean([math.sqrt(r['covariance'][1][1]) for r in rows])))


def worker(output, dataset, start):
    manifest = json.loads((output/'manifest.json').read_text())
    source, calibration = Path(manifest['source']), Path(manifest['calibration'])
    config = manifest['datasets'][str(dataset)]
    index = next(i for i, (a, _) in enumerate(config['windows']) if a == start)
    original = source/f'd{dataset}_w{index:03d}'
    folder = output/f'd{dataset}_t{start}'
    folder.mkdir()
    package = json.loads((original/'nominal.json').read_text())
    events = [json.loads(s) for s in (calibration/f'd{dataset}/events.jsonl').read_text().splitlines()]
    points = json.loads((output/f'd{dataset}_information.json').read_text())['log_scale_points']
    binary = manifest['binary']['path']
    predictions, commands = [], []
    residuals = []
    for index, point in enumerate(points):
        scales = np.exp(point)
        weighted = apply_scales(package, scales, config['first_timestamp_s']+config['cutoff_elapsed_s'], 0.)
        path = folder/f'p{index:02d}.json'
        write_json(path, weighted)
        native = folder/f'p{index:02d}'
        command = [binary, str(path), str(native), 'central_lm', '100', '--linearization']
        begun = time.perf_counter()
        with (folder/f'p{index:02d}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        commands.append(dict(command=command, exit_code=result.returncode, seconds=time.perf_counter()-begun))
        write_json(folder/'commands.json', commands)
        if result.returncode:
            raise RuntimeError('native uncertainty solve failed')
        prediction = forecast(weighted, read_rows(native/'final_estimates.csv'),
            json.loads((native/'linearization.json').read_text()), events,
            config['first_timestamp_s']+start, 5., scales)
        write_json(native/'forecast.json', prediction)
        predictions.append(prediction['predictions'])
        residuals.append(prediction['covariance_solve_relative_residual'])
    mixture = mix_predictions(predictions)
    write_json(folder/'mixture.json', mixture)
    rows = {'mixture':mixture}
    for arm in ('nominal','joint_qr'):
        rows[arm] = json.loads((original/arm/'forecast.json').read_text())['predictions']
        rescore(mixture, rows[arm])
    selections = [('all', lambda r: True)]
    selections += [(f'robot{r}', lambda row, r=r: row['robot']==r) for r in range(1,6)]
    if dataset == 1 and start == 50:
        selections += [(f'robot5_subject{s}', lambda r, s=s: r['robot']==5 and r['subject']==s)
                       for s in (6,7)]
    summary = [dict(dataset=dataset, start=start, group=group, arm=arm, **statistics(selected))
        for group, select in selections for arm, values in rows.items()
        if (selected := [r for r in values if select(r)])]
    write_json(folder/'summary.json', dict(records=summary, covariance_solve_residuals=residuals,
        complete=True, component_count=len(points), truth_used=False))


def run(source, calibration, output):
    source, calibration, output = source.resolve(), calibration.resolve(), output.resolve()
    hashes = {}
    for directory in (source, calibration):
        hashes[str(directory)] = {}
        for line in (directory/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            if sha256(directory/name) != digest:
                raise ValueError('changed source archive: '+name)
            hashes[str(directory)][name] = digest
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    binary = root/'build/native/bin/native-joint-audit'
    shutil.copy2(binary, output/'native.binary')
    for name in ('experiments/audit_noise_parameter_uncertainty.py',
                 'tests/test_noise_parameter_uncertainty.py',
                 'experiments/network_noise_prediction.py','experiments/network_noise_support.py'):
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    old = json.loads((source/'manifest.json').read_text())
    manifest = dict(source=str(source), calibration=str(calibration), source_hashes=hashes,
        binary=dict(path=str(binary), sha256=sha256(binary)), datasets=old['datasets'],
        all_terminal=False, truth_used=False, runs=[],
        scope='Six selected discovery cases, local expected-Fisher log-scale sampling uncertainty; not a Bayesian posterior, sandwich covariance, online method or full-suffix test.')
    write_json(output/'manifest.json', manifest)
    for dataset in (1,4):
        fit_dir = calibration/f'd{dataset}/fits/joint_qr'
        fit = json.loads((fit_dir/'fit.json').read_text())
        with np.load(fit_dir/'contrast.npz') as data:
            covariance, information = parameter_covariance(data['base'], data['components'], fit['scales'])
        eigen_error = float(np.max(np.abs(np.linalg.eigvalsh(information)-fit['fisher_eigenvalues'])))
        correlation = information/np.sqrt(np.outer(np.diag(information), np.diag(information)))
        correlation_error = float(np.max(np.abs(correlation-fit['fisher_correlation'])))
        if eigen_error > 1e-8 or correlation_error > 1e-10:
            raise ValueError('Fisher information does not match original fit')
        write_json(output/f'd{dataset}_information.json', dict(covariance=covariance.tolist(),
            information=information.tolist(), log_scale_sd=np.sqrt(np.diag(covariance)).tolist(),
            log_scale_points=cubature(np.log(fit['scales']), covariance).tolist(),
            fisher_eigenvalue_error=eigen_error, fisher_correlation_error=correlation_error))
    # Workers have a total deadline, not ten independent unbounded solve budgets.
    active = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for dataset in (1,4):
        for start in (30,50,85):
            command = ['timeout','240',sys.executable,'-m','experiments.audit_noise_parameter_uncertainty', 'worker',
                       '--output',str(output),'--dataset',str(dataset),'--start',str(start)]
            log = (output/f'd{dataset}_t{start}.log').open('x')
            active.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env),
                           log, command, time.perf_counter()))
    for process, log, command, begun in active:
        code = process.wait()
        log.close()
        manifest['runs'].append(dict(command=command, exit_code=code, elapsed_until_collected_s=time.perf_counter()-begun))
        write_json(output/'manifest.json', manifest)
    manifest['all_terminal'] = True
    write_json(output/'manifest.json', manifest)
    if any(r['exit_code'] for r in manifest['runs']):
        raise RuntimeError('incomplete diagnostic; failures retained')
    records = []
    for folder in sorted(output.glob('d*_t*')):
        if folder.is_dir():
            records += json.loads((folder/'summary.json').read_text())['records']
    write_json(output/'summary.json', dict(records=records, scope=manifest['scope']))
    for row in records:
        if row['group']=='all' or row['group']=='robot5_subject6':
            print(json.dumps(row), flush=True)


def verify(output):
    from experiments.analyze_rolling_prediction import check_prediction, check_weights
    manifest = json.loads((output/'manifest.json').read_text())
    if not manifest['all_terminal'] or any(r['exit_code'] for r in manifest['runs']):
        raise ValueError('incomplete experiment')
    for directory, files in manifest['source_hashes'].items():
        for name, digest in files.items():
            if sha256(Path(directory)/name) != digest:
                raise ValueError('source changed')
    for path in (Path(manifest['binary']['path']), output/'native.binary'):
        if sha256(path) != manifest['binary']['sha256']:
            raise ValueError('binary changed')
    maximum = dict(weight=0., innovation=0., energy=0., nll=0., mixture_nll=0., covariance_residual=0.)
    solves, count = [], 0
    for dataset in (1,4):
        config = manifest['datasets'][str(dataset)]
        events = [json.loads(s) for s in (Path(manifest['calibration'])/f'd{dataset}/events.jsonl').read_text().splitlines()]
        lookup = {(e['robot'],e['source_row']):e for e in events if e['kind']=='observation'}
        information = json.loads((output/f'd{dataset}_information.json').read_text())
        points = information['log_scale_points']
        delta = np.array(points)-np.log(config['scales'])
        if not np.allclose(delta.T@delta/len(points), information['covariance'], atol=1e-12, rtol=1e-12):
            raise ValueError('cubature covariance mismatch')
        for start in (30,50,85):
            index = next(i for i,(a,_) in enumerate(config['windows']) if a==start)
            nominal = json.loads((Path(manifest['source'])/f'd{dataset}_w{index:03d}/nominal.json').read_text())
            folder = output/f'd{dataset}_t{start}'
            components = []
            for k, point in enumerate(points):
                package = json.loads((folder/f'p{k:02d}.json').read_text())
                maximum['weight'] = max(maximum['weight'], check_weights(nominal, package,
                    np.exp(point), config['first_timestamp_s']+config['cutoff_elapsed_s']))
                if any(p['timestamp_s'] > config['first_timestamp_s']+start
                       for r in package['robots'] for p in r['poses']):
                    raise ValueError('future state support')
                prediction = json.loads((folder/f'p{k:02d}/forecast.json').read_text())
                maximum['covariance_residual'] = max(maximum['covariance_residual'], prediction['covariance_solve_relative_residual'])
                components.append(prediction['predictions'])
                for row in components[-1]:
                    event = lookup[row['robot'],row['source_row']]
                    if identity(row) != (event['robot'],event['source_row'],event['timestamp_s'],event['landmark_subject']):
                        raise ValueError('forecast identity mismatch')
                    for key, value in check_prediction(row, event).items():
                        maximum[key] = max(maximum[key], value)
                    count += 1
                solves.append(json.loads((folder/f'p{k:02d}/summary.json').read_text()))
            mixtures = json.loads((folder/'mixture.json').read_text())
            for row, values in zip(mixtures, zip(*components)):
                expected = -np.logaddexp.reduce([-r['nll'] for r in values])+math.log(len(values))
                maximum['mixture_nll'] = max(maximum['mixture_nll'], abs(expected-row['nll']))
    if maximum['mixture_nll'] > 1e-10:
        raise ValueError('mixture score mismatch')
    result = dict(maximum_errors=maximum, component_predictions_checked=count,
        nonlinear_solves=len(solves), max_lm_iterations=max(r['lm_iterations'] for r in solves),
        max_posthoc_gradient=max(r['posthoc_gradient_norm'] for r in solves), truth_used=False)
    write_json(output/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run','worker','verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('results/dranppr_rolling_prediction_20261028'))
    parser.add_argument('--calibration', type=Path, default=Path('results/dranppr_network_noise_transfer_qr_20261003'))
    parser.add_argument('--dataset', type=int)
    parser.add_argument('--start', type=int)
    args = parser.parse_args()
    if args.mode=='run':
        run(args.source, args.calibration, args.output)
    elif args.mode=='worker':
        worker(args.output, args.dataset, args.start)
    else:
        verify(args.output)
