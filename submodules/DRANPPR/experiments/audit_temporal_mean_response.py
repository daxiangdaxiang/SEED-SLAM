"""Past-only first-order mean-response forecasts with no backend changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.analyze_network_noise_transfer import moments, rescore
from experiments.audit_mean_evidence import sha256, write_json
from experiments.evaluate_incremental_reference import read_rows
from experiments.mean_response import fit_native_response, retracted_means
from experiments.network_noise_prediction import forecast


def parity(candidate, reference):
    a, b = candidate['predictions'], reference['predictions']
    rescore(a, b)
    if candidate['unknown_object_observations'] != reference['unknown_object_observations']:
        raise ValueError('unknown-object support changed')
    errors = {key: max(float(np.max(np.abs(np.asarray(x[key])-y[key])))
                       for x, y in zip(a, b)) for key in ('mean', 'covariance', 'nll')}
    if max(errors.values()) > 1e-8:
        raise ValueError('original forecast parity failed: '+str(errors))
    return errors


def worker(source, output, dataset, elapsed, law):
    folder = output/f'd{dataset}_t{elapsed}_{law}'
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = source/f'd{dataset}/t{elapsed}_{law}'
    package = json.loads((checkpoint.with_suffix('.json')).read_text())
    linear = json.loads((checkpoint/'linearization.json').read_text())
    states = read_rows(checkpoint/'final_estimates.csv')
    saved = json.loads((checkpoint/'forecast.json').read_text())
    events = [json.loads(s) for s in (source/f'd{dataset}/events.jsonl').read_text().splitlines()]
    scales = np.ones(5) if law == 'nominal' else json.loads(
        (source/f'd{dataset}/fits/joint_qr/fit.json').read_text())['scales']
    started = time.perf_counter()
    original = forecast(package, states, linear, events, saved['checkpoint_s'], 5., scales)
    comparison = parity(original, saved)
    response = fit_native_response(linear)
    delta = response['step']+response['sensitivity']@response['mean']
    pose_shift, yaw_shift, object_shift = [], [], []
    for key, indices in response['columns'].items():
        change = delta[indices]
        if key[0] == 'x':
            pose_shift.append(float(np.linalg.norm(change[:2])))
            yaw_shift.append(float(abs(change[2])))
        else:
            object_shift.append(float(np.linalg.norm(change)))
    fit = {k: response[k] for k in ('orthogonality_relative', 'original_linear_loss',
                                    'adjusted_linear_loss', 'stationarity_relative')}
    fit.update(robots=response['robots'], mean_m=response['mean'].tolist(),
        mean_sd_m=np.sqrt(np.diag(response['covariance'])).tolist(),
        mean_covariance=response['covariance'].tolist(),
        singular_values=response['singular_values'].tolist(),
        max_pose_translation_step_m=max(pose_shift), max_yaw_step_rad=max(yaw_shift),
        max_object_step_m=max(object_shift), max_base_gn_step=float(np.max(np.abs(response['step']))),
        source_linear_sha256=sha256(checkpoint/'linearization.json'),
        future_used=False, truth_used=False, fixed_chart=True)
    write_json(folder/'fit.json', fit)
    full = forecast(package, states, linear, events, saved['checkpoint_s'], 5., scales,
                    mean_response=response)
    mean_scores = rescore(full['predictions'], original['predictions'])
    mean_only = [dict(p, covariance=base['covariance'], **score)
                 for p, base, score in zip(full['predictions'], original['predictions'], mean_scores)]
    for p in mean_only:
        p['mean_covariance_addition'] = [[0., 0.], [0., 0.]]
    for name, rows in [('original', original['predictions']), ('mean_only', mean_only),
                       ('mean_uncertainty', full['predictions'])]:
        write_json(folder/(name+'.json'), dict(predictions=rows, dataset=dataset,
            elapsed=elapsed, law=law, arm=name, unknown_object_observations=original['unknown_object_observations'],
            checkpoint_s=saved['checkpoint_s'], horizon_s=5., truth_used=False,
            negative_predicted_ranges=sum(p['mean'][1] <= 0. for p in rows)))
    write_json(folder/'verification.json', dict(original_parity=comparison,
        worker_seconds=time.perf_counter()-started,
        covariance_solve_relative_residual=full['covariance_solve_relative_residual']))


def summarize(source):
    groups = []
    for dataset in (1, 4):
        for law in ('nominal', 'joint_qr'):
            for arm in ('original', 'mean_only', 'mean_uncertainty'):
                rows = []
                for elapsed in (30, 60, 90):
                    data = json.loads((source/f'd{dataset}_t{elapsed}_{law}/{arm}.json').read_text())
                    rows.extend(dict(p, elapsed=elapsed) for p in data['predictions'])
                selections = [('dataset', None, rows)]
                for field in ('elapsed', 'robot', 'subject'):
                    selections += [(field, value, [p for p in rows if p[field] == value])
                                   for value in sorted({p[field] for p in rows})]
                for family, value, selected in selections:
                    ranges = np.array([p['innovation'][1] for p in selected])
                    variance = np.array([p['covariance'][1][1] for p in selected])
                    extra = np.array([p.get('mean_covariance_addition', [[0, 0], [0, 0]])[1][1] for p in selected])
                    groups.append(dict(dataset=dataset, law=law, arm=arm, family=family, value=value,
                        **moments(selected), range_rmse_m=float(np.sqrt(np.mean(ranges**2))),
                        range_nll=float(np.mean(.5*(np.log(2*np.pi*variance)+ranges**2/variance))),
                        mean_range_variance=float(np.mean(variance)),
                        mean_range_parameter_variance=float(np.mean(extra)),
                        negative_predicted_ranges=sum(p['mean'][1] <= 0. for p in selected)))
    return groups


def run(source, output):
    source, output = source.resolve(), output.resolve()
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(source/name) != digest:
            raise ValueError('source archive changed: '+name)
        hashes[name] = digest
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for name in ['experiments/audit_temporal_mean_response.py', 'experiments/mean_response.py',
                 'experiments/network_noise_prediction.py', 'experiments/audit_mean_evidence.py',
                 'experiments/analyze_network_noise_transfer.py', 'tests/test_temporal_mean_response.py',
                 'docs/superpowers/plans/2026-10-27-temporal-mean-response.md']:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    manifest = dict(source=str(source), source_hashes=hashes, all_workers_terminal=False,
        truth_used=False, runs=[], scope='Past-only fixed-chart response, previously inspected windows, no estimator change or online claim')
    write_json(output/'manifest.json', manifest)
    for dataset in (1, 4):
        for elapsed in (30, 60, 90):
            for law in ('nominal', 'joint_qr'):
                command = [sys.executable, '-m', 'experiments.audit_temporal_mean_response',
                    '--source', str(source), '--output', str(output), '--dataset', str(dataset),
                    '--elapsed', str(elapsed), '--law', law]
                tag = f'd{dataset}_t{elapsed}_{law}'
                started = time.perf_counter()
                with (output/(tag+'.log')).open('x') as log:
                    try:
                        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240,
                            env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')).returncode
                    except subprocess.TimeoutExpired:
                        code = 'timeout_240s'
                record = dict(dataset=dataset, elapsed=elapsed, law=law, command=command,
                    exit_code=code, wall_seconds=time.perf_counter()-started)
                manifest['runs'].append(record)
                write_json(output/'manifest.json', manifest)
                print(json.dumps(record), flush=True)
                if code != 0:
                    raise RuntimeError('worker failed; retained without retuning')
    write_json(output/'summary.json', summarize(output))
    manifest['all_workers_terminal'] = True
    write_json(output/'manifest.json', manifest)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))


def retraction_audit(source, primary, output):
    if not json.loads((primary/'manifest.json').read_text())['all_workers_terminal']:
        raise ValueError('incomplete primary experiment')
    for archive in (source, primary):
        for line in (archive/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            if sha256(archive/name) != digest:
                raise ValueError('changed retraction audit input: '+name)
    output.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('mean_response.py')):
        shutil.copy2(path, output/path.name)
    records, summary = [], []
    for dataset in (1, 4):
        for law in ('nominal', 'joint_qr'):
            combined, linear_rows, original_rows = [], [], []
            for elapsed in (30, 60, 90):
                started = time.perf_counter()
                tag = f'd{dataset}_t{elapsed}_{law}'
                checkpoint = source/f'd{dataset}/t{elapsed}_{law}'
                data = json.loads((primary/tag/'mean_uncertainty.json').read_text())
                expected = json.loads((primary/tag/'fit.json').read_text())
                linear = json.loads((checkpoint/'linearization.json').read_text())
                response = fit_native_response(linear)
                fit_difference = float(np.max(np.abs(response['mean']-expected['mean_m'])))
                if fit_difference > 1e-10:
                    raise ValueError('fixed mean refit parity failed')
                events = [json.loads(s) for s in (source/f'd{dataset}/events.jsonl').read_text().splitlines()]
                predicted = retracted_means(json.loads(checkpoint.with_suffix('.json').read_text()),
                    read_rows(checkpoint/'final_estimates.csv'), events, data['checkpoint_s'], 5., response)
                if len(predicted) != len(data['predictions']):
                    raise ValueError('retraction support differs')
                updated, errors = [], []
                for mean, reference in zip(predicted, data['predictions']):
                    identity = ('robot', 'subject', 'timestamp_s', 'source_row')
                    if any(mean[k] != reference[k] for k in identity):
                        raise ValueError('retraction identity changed')
                    change = np.array(mean['mean'])-reference['mean']
                    change[0] = np.arctan2(np.sin(change[0]), np.cos(change[0]))
                    innovation = np.array(reference['innovation'])-change
                    innovation[0] = np.arctan2(np.sin(innovation[0]), np.cos(innovation[0]))
                    updated.append(dict(reference, mean=mean['mean'], innovation=innovation.tolist()))
                    errors.append(change)
                for row, score in zip(updated, rescore(updated, data['predictions'])):
                    row.update(score)
                write_json(output/(tag+'.json'), dict(predictions=updated,
                    scope='Nonlinear evaluation of SAME linear step, FIXED original response covariance; not a nonlinear optimized or calibrated arm'))
                error = np.asarray(errors)
                record = dict(dataset=dataset, law=law, elapsed=elapsed, fit_max_difference=fit_difference,
                    max_bearing_mean_difference_rad=float(np.max(np.abs(error[:, 0]))),
                    max_range_mean_difference_m=float(np.max(np.abs(error[:, 1]))),
                    range_mean_difference_rms_m=float(np.sqrt(np.mean(error[:, 1]**2))),
                    nonlinear_mean_nll=moments(updated)['mean_nll'],
                    linear_mean_nll=moments(data['predictions'])['mean_nll'],
                    seconds=time.perf_counter()-started)
                records.append(record)
                write_json(output/'records.json', records)
                print(json.dumps(record), flush=True)
                combined.extend(updated)
                linear_rows.extend(data['predictions'])
                original_rows.extend(json.loads((primary/tag/'original.json').read_text())['predictions'])
            summary.append(dict(dataset=dataset, law=law, original=moments(original_rows),
                linear=moments(linear_rows), nonlinear_mean_fixed_covariance=moments(combined)))
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', dict(source=str(source.resolve()), primary=str(primary.resolve()),
        post_selected_approximation_audit=True, truth_used=False, complete=True,
        optimized_nonlinearly=False, covariance_relinearized=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=int, choices=[1, 4])
    parser.add_argument('--elapsed', type=int, choices=[30, 60, 90])
    parser.add_argument('--law', choices=['nominal', 'joint_qr'])
    parser.add_argument('--retraction-source', type=Path)
    args = parser.parse_args()
    if args.retraction_source is not None:
        retraction_audit(args.source, args.retraction_source, args.output)
    elif args.dataset is None:
        run(args.source, args.output)
    else:
        worker(args.source, args.output, args.dataset, args.elapsed, args.law)
