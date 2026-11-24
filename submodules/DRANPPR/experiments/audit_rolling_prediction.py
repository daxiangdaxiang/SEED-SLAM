"""Complete, disjoint past-only forecast coverage of the 120 s D1/D4 benchmark."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.analyze_network_noise_transfer import moments, rescore
from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_temporal_mean_response import parity
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_prediction import forecast
from experiments.network_noise_support import apply_scales
from experiments.prepare_frozen_completion import prepare

ARMS = ('nominal', 'joint_qr')
LINKS = ['1:2', '2:3', '3:4', '4:5']


def windows(start, end, step):
    if not all(map(math.isfinite, (start, end, step))) or step <= 0 or start >= end:
        raise ValueError('invalid rolling schedule')
    boundaries = [float(start)]
    tick = math.floor(start/step)+1
    while tick*step < end:
        boundaries.append(float(tick*step))
        tick += 1
    boundaries.append(float(end))
    return list(zip(boundaries[:-1], boundaries[1:]))


def identity(row):
    return row['robot'], row['source_row'], row['timestamp_s'], row['subject']


def support_ledger(package, events, start, end, rows):
    local = {(r['id'], o['label']) for r in package['robots'] for o in r['objects']}
    global_labels = {s for _, s in local}
    ledger, expected = [], []
    for e in events:
        if e['kind'] != 'observation' or not start < e['timestamp_s'] <= end:
            continue
        if e['available_time_s'] != e['timestamp_s'] or e['support_max_time_s'] > e['timestamp_s']:
            raise ValueError('noncausal observation support')
        known = (e['robot'], e['landmark_subject']) in local
        status = ('scored' if known else 'no_local_object_copy'
                  if e['landmark_subject'] in global_labels else 'unseen_object')
        record = dict(robot=e['robot'], source_row=e['source_row'],
                      timestamp_s=e['timestamp_s'], subject=e['landmark_subject'], status=status)
        ledger.append(record)
        if known:
            expected.append(identity(record))
    actual = [identity(r) for r in rows]
    if len(set(actual)) != len(actual) or actual != expected:
        raise ValueError('missing, duplicate or spurious predictive support')
    if len({identity(r) for r in ledger}) != len(ledger):
        raise ValueError('duplicate observation support')
    return ledger


def statistics(rows):
    if not rows:
        return dict(count=0, frame_count=0)
    frames, periods = defaultdict(list), defaultdict(list)
    for row in rows:
        frames[row['robot'], row['timestamp_s']].append(row['nll'])
        periods[row['window']].append(row['nll'])
    errors = np.array([r['innovation'] for r in rows])
    variance = np.array([np.diag(r['covariance']) for r in rows])
    return dict(**moments(rows), frame_count=len(frames),
        frame_equal_nll=float(np.mean([np.mean(v) for v in frames.values()])),
        window_equal_nll=float(np.mean([np.mean(v) for v in periods.values()])),
        range_rmse_m=float(np.sqrt(np.mean(errors[:, 1]**2))),
        bearing_rmse_rad=float(np.sqrt(np.mean(errors[:, 0]**2))),
        range_nll=float(np.mean(.5*(np.log(2*np.pi*variance[:, 1])+errors[:, 1]**2/variance[:, 1]))),
        bearing_nll=float(np.mean(.5*(np.log(2*np.pi*variance[:, 0])+errors[:, 0]**2/variance[:, 0]))))


def execute(command, log_path, records):
    started = time.perf_counter()
    with log_path.open('x') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout_180s'
    records.append(dict(command=command, exit_code=code, seconds=time.perf_counter()-started))
    write_json(log_path.parent/'commands.json', records)
    if code != 0:
        raise RuntimeError('native run failed; evidence retained without retry')


def worker(source, output, dataset, index):
    config = json.loads((output/'manifest.json').read_text())['datasets'][str(dataset)]
    start, end = config['windows'][index]
    case = source/f'd{dataset}'
    folder = output/f'd{dataset}_w{index:03d}'
    folder.mkdir(exist_ok=False)
    events = [json.loads(s) for s in (case/'events.jsonl').read_text().splitlines()]
    warm = json.loads((case/'warm_start.json').read_text())
    first = config['first_timestamp_s']
    root = Path(__file__).resolve().parents[1]
    binary = root/'build/native/bin'
    commands = []
    if index == 0:
        package = warm
    else:
        execute([str(binary/'native-joint-replay'), str(case/'events.jsonl'), str(folder/'replay'),
            'joint_cached_one', str(start), '--warm-start', str(case/'warm_start.json'),
            '--lazy-refresh', *LINKS], folder/'replay.log', commands)
        package = prepare(events, read_rows(folder/'replay/final_estimates.csv'), warm, start, 0.)
    arms, checks, ledgers = {}, {}, {}
    for arm in ARMS:
        scales = [1.]*5 if arm == 'nominal' else config['scales']
        weighted = package if index == 0 else apply_scales(package, scales, first+config['cutoff_elapsed_s'], 0.)
        path = folder/(arm+'.json')
        write_json(path, weighted)
        native = folder/arm
        execute([str(binary/'native-joint-audit'), str(path), str(native),
                 'central_lm', '100', '--linearization'], folder/(arm+'.log'), commands)
        result = forecast(weighted, read_rows(native/'final_estimates.csv'),
            json.loads((native/'linearization.json').read_text()), events,
            first+start, end-start, scales)
        ledger = support_ledger(weighted, events, first+start, first+end, result['predictions'])
        if sum(r['status'] != 'scored' for r in ledger) != result['unknown_object_observations']:
            raise ValueError('unsupported-observation accounting mismatch')
        if start in (30., 60., 90.):
            old = json.loads((case/f't{int(start)}_{arm}/forecast.json').read_text())
            checks[arm] = parity(result, old)
        result.update(dataset=dataset, arm=arm, window=index, start_elapsed_s=start, end_elapsed_s=end)
        write_json(native/'forecast.json', result)
        arms[arm] = result
        ledgers[arm] = ledger
    rescore(arms['nominal']['predictions'], arms['joint_qr']['predictions'])
    if ledgers['nominal'] != ledgers['joint_qr']:
        raise ValueError('arm support ledger mismatch')
    write_json(folder/'ledger.json', ledgers['nominal'])
    write_json(folder/'verification.json', dict(old_window_parity=checks,
        covariance_residuals={a:arms[a]['covariance_solve_relative_residual'] for a in ARMS},
        input_observations=len(ledgers['nominal']),
        scored_observations=len(arms['nominal']['predictions']), complete=True))


def summarize(output):
    manifest = json.loads((output/'manifest.json').read_text())
    groups, coverage, comparisons = [], [], []
    for dataset in (1, 4):
        config = manifest['datasets'][str(dataset)]
        all_rows, ledger = {arm:[] for arm in ARMS}, []
        for index, (start, end) in enumerate(config['windows']):
            folder = output/f'd{dataset}_w{index:03d}'
            ledger += json.loads((folder/'ledger.json').read_text())
            for arm in ARMS:
                data = json.loads((folder/arm/'forecast.json').read_text())
                all_rows[arm] += [dict(r, window=index, start_elapsed_s=start,
                    band=min(3, int((r['timestamp_s']-config['first_timestamp_s'])//30)))
                    for r in data['predictions']]
        actual = [identity(r) for r in ledger]
        expected = [tuple(r) for r in config['expected_observation_keys']]
        if len(set(actual)) != len(actual) or actual != expected:
            raise ValueError('suffix event accounting is not complete and disjoint')
        statuses = dict(Counter(r['status'] for r in ledger))
        coverage.append(dict(dataset=dataset, observations=len(ledger), statuses=statuses,
            scored_fraction=statuses.get('scored', 0)/len(ledger), windows=len(config['windows']),
            start_elapsed_s=config['cutoff_elapsed_s'], end_elapsed_s=120.,
            by_robot=[dict(robot=r, **dict(Counter(x['status'] for x in ledger if x['robot']==r)))
                      for r in range(1, 6)]))
        rescore(all_rows['nominal'], all_rows['joint_qr'])
        for arm, rows in all_rows.items():
            selections = [('dataset', None, rows)]
            for field, values in [('robot', range(1, 6)), ('window', range(len(config['windows']))),
                                  ('band', range(4))]:
                selections += [(field, value, [p for p in rows if p[field]==value]) for value in values]
            for family, value, selected in selections:
                groups.append(dict(dataset=dataset, arm=arm, family=family, value=value, **statistics(selected)))
        for index, (start, end) in enumerate(config['windows']):
            nominal = [r for r in all_rows['nominal'] if r['window']==index]
            adapted = [r for r in all_rows['joint_qr'] if r['window']==index]
            comparisons.append(dict(dataset=dataset, window=index, start=start, end=end,
                count=len(nominal), nll_difference=float(np.mean([b['nll']-a['nll']
                    for a,b in zip(nominal, adapted)])) if nominal else None))
        for arm in ARMS:
            write_json(output/f'd{dataset}_{arm}_predictions.json', all_rows[arm])
    result = dict(groups=groups, coverage=coverage, windows=comparisons, truth_used=False,
        scope='Descriptive dependent marginal scores over complete declared 120 s benchmark suffix; not full raw sequence, not ATE, not online deployment')
    write_json(output/'summary.json', result)
    return result


def run(source, output):
    source, output = source.resolve(), output.resolve()
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(source/name) != digest:
            raise ValueError('source archive changed: '+name)
        hashes[name] = digest
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for dataset in (1, 4):
        case = source/f'd{dataset}'
        warm = json.loads((case/'warm_start.json').read_text())
        events = [json.loads(s) for s in (case/'events.jsonl').read_text().splitlines()]
        first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
        metadata = json.loads((case/'metadata.json').read_text())
        fit = json.loads((case/'fits/joint_qr/fit.json').read_text())
        if (warm.get('truth_used', False) or fit.get('truth_used', False) or
                sha256(case/'events.jsonl') != metadata['event_sha256'] or
                fit['source_sha256'] != sha256(case/'training/linearization.json')):
            raise ValueError('invalid causal source provenance')
        schedule = windows(cutoff-first, 120., 5.)
        expected = [(e['robot'], e['source_row'], e['timestamp_s'], e['landmark_subject'])
                    for e in events if e['kind']=='observation' and cutoff < e['timestamp_s'] <= first+120.]
        datasets[str(dataset)] = dict(first_timestamp_s=first, cutoff_elapsed_s=cutoff-first,
            windows=schedule, scales=fit['scales'], expected_observation_keys=expected,
            raw_duration_s=metadata['last_event_time_s']-first)
    binary = root/'build/native/bin'
    libraries = subprocess.check_output(['ldd', str(binary/'native-joint-audit')], text=True)
    linked = [Path(s.split('=>',1)[1].split()[0]) for s in libraries.splitlines() if '=>' in s]
    if str(root/'build/research-deps/gtsam-imesa-build/gtsam/libgtsam') not in libraries or 'libtbb' in libraries:
        raise ValueError('unexpected native runtime dependencies')
    for name in ['experiments/audit_rolling_prediction.py', 'experiments/network_noise_prediction.py',
                 'experiments/prepare_frozen_completion.py', 'experiments/network_noise_support.py',
                 'experiments/audit_temporal_mean_response.py', 'experiments/analyze_network_noise_transfer.py',
                 'tests/test_rolling_prediction.py', 'docs/superpowers/plans/2026-10-28-rolling-prediction.md']:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    manifest = dict(source=str(source), source_hashes=hashes, datasets=datasets, runs=[],
        all_workers_terminal=False, truth_used=False, arms=ARMS, libraries=libraries,
        library_hashes={str(p):sha256(p) for p in linked if p.is_file()},
        binary_hashes={str(binary/n):sha256(binary/n) for n in ('native-joint-audit','native-joint-replay')},
        worker_timeout_s=240, scope='Frozen-scale causal batch diagnostic, original 120 s benchmark only')
    write_json(output/'manifest.json', manifest)
    for dataset in (1, 4):
        for index in range(len(datasets[str(dataset)]['windows'])):
            command = [sys.executable, '-m', 'experiments.audit_rolling_prediction',
                '--source', str(source), '--output', str(output), '--dataset', str(dataset), '--index', str(index)]
            started = time.perf_counter()
            with (output/f'd{dataset}_w{index:03d}.log').open('x') as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')).returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout_240s'
            record = dict(dataset=dataset, window=index, command=command, exit_code=code,
                          seconds=time.perf_counter()-started)
            manifest['runs'].append(record)
            write_json(output/'manifest.json', manifest)
            print(json.dumps(record), flush=True)
            if code != 0:
                raise RuntimeError('worker failed; preserve failed window, no parameter retry')
    summarize(output)
    manifest['all_workers_terminal'] = True
    write_json(output/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=int, choices=[1, 4])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.dataset is None:
        run(args.source, args.output)
    else:
        worker(args.source, args.output, args.dataset, args.index)
