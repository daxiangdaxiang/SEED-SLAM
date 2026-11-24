"""Persistent native suffix replay, fixed opportunities and causal byte ceilings."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_native_matched import dump, sha
from experiments.evaluate_incremental_reference import read_rows


def verify(folder, package, reference, credits=None):
    s = json.loads((folder/'summary.json').read_text())
    if not math.isclose(s['initial_cost'], reference['initial_cost'], abs_tol=1e-8, rel_tol=1e-12):
        raise ValueError('initial physical graph mismatch')
    for key in ['frames', 'packets', 'physical_factors', 'prefix_physical_factors']:
        if s[key] != reference[key]:
            raise ValueError(f'{key} mismatch')
    if s['root_displacement'] > 1e-8:
        raise ValueError('root gauge changed')
    updates = read_rows(folder/'updates.csv')
    keys = [(int(r['robot']), float(r['timestamp_s'])) for r in updates]
    if keys != [(r['robot'], r['timestamp_s']) for r in credits]:
        raise ValueError('different causal output frames')
    if s['mode'].startswith('joint'):
        if sum(int(r['optimizer_bytes'])+int(r['control_bytes']) for r in updates) != s['application_bytes']:
            raise ValueError('DRAN byte ledger mismatch')
        total, messages = 0, 0
        for line in (folder/'control_messages.jsonl').open():
            row = json.loads(line)
            total += row['application_bytes']
            messages += 1
        if total != s['control_application_bytes'] or messages != s['control_messages']:
            raise ValueError('DRAN control ledger mismatch')
    else:
        edges = {frozenset(e) for e in package['links']}
        total, payload, messages = 0, 0, 0
        for line in (folder/'messages.jsonl').open():
            row = json.loads(line)
            if frozenset((row['from'], row['to'])) not in edges:
                raise ValueError('message outside topology')
            if row['application_bytes'] != row['payload_bytes']+40:
                raise ValueError('invalid message header')
            total += row['application_bytes']
            payload += row['payload_bytes']
            messages += 1
        if (total, payload, messages) != (s['application_bytes'], s['payload_bytes'], s['messages']):
            raise ValueError('baseline byte ledger mismatch')
        if messages != 4*len(edges)*s['rounds']:
            raise ValueError('incomplete handshake')
        trace = [json.loads(line) for line in (folder/'trace.jsonl').open()]
        if len(trace) != len(keys) or trace[-1]['application_bytes'] != total:
            raise ValueError('trace frame mismatch')
        if 'byte-credit' in s['schedule']:
            if any(r['application_bytes'] > c['application_bytes'] for r, c in zip(trace, credits)):
                raise ValueError('causal communication ceiling exceeded')
    s.update(ledger_verified=True, information_verified=True)
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--dran-mode', choices=['joint_cached', 'joint_one'], default='joint_cached')
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    if package.get('truth_used', False):
        raise ValueError('truth-assisted package')
    if 'event_sha256' in package.get('provenance', {}):
        if package['provenance']['event_sha256'] != sha(args.events):
            raise ValueError('prefix/event provenance hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest = dict(truth_used_by_estimator=False, duration=args.duration, runs=[],
        package_sha256=sha(args.package), event_sha256=sha(args.events), binaries={}, sources={},
        all_estimators_terminal=False, dran_mode=args.dran_mode,
        scope='Common primal past, cold internal solvers; persistent chronological suffix; one process per method, sequential runs',
        protocol='DRAN one GN step/frame, declared existing mode; baselines one sweep/frame and causal cumulative DRAN byte-credit replay. Credit schedule depends only on arrived DRAN data, not GT. Not equal compute or deployed scheduling.')
    shutil.copy2(args.package, args.output/'input.json')
    shutil.copy2(args.events, args.output/'events.jsonl')
    binaries = dict(dran=Path('/tmp/dranppr_backend_build/bin/native-joint-replay'),
                    baseline=Path('/tmp/dranppr_matched_build/native-matched-baseline'))
    for tag, path in binaries.items():
        linked = subprocess.check_output(['ldd', str(path)], text=True)
        libraries = [Path(s.split('=>', 1)[1].split()[0]) for s in linked.splitlines() if '=>' in s]
        manifest['binaries'][tag] = dict(path=str(path), sha256=sha(path), ldd=linked,
            library_sha256={str(p): sha(p) for p in libraries if p.is_file()})
        shutil.copy2(path, args.output/f'{tag}.binary')
    cache = Path('/tmp/dranppr_matched_build/CMakeCache.txt')
    shutil.copy2(cache, args.output/'baseline_CMakeCache.txt')
    config = {}
    for line in cache.read_text().splitlines():
        if '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            config[key.split(':', 1)[0]] = value
    manifest['upstream'] = {}
    for key in ['IMESA_SOURCE_DIR', 'RIMESA_SOURCE_DIR', 'RISAM_SOURCE_DIR']:
        source = Path(config[key])
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True)
        manifest['upstream'][key] = dict(path=str(source), commit=commit, status=status)
    manifest['static_libraries'] = {key: dict(path=config[key], sha256=sha(config[key]))
        for key in ['RIMESA_LIBRARY', 'RISAM_LIBRARY']}
    names = ['experiments/audit_native_online.py', 'experiments/online_stream.h',
        'experiments/native_matched/baseline.cpp', 'experiments/native_matched/online.h',
        'experiments/native_matched/CMakeLists.txt', 'experiments/native_local/joint_package.h',
        'experiments/native_local/joint_step.h', 'experiments/native_local/joint_replay.cpp',
        'tests/test_native_matched_online.py', 'tests/test_audit_native_online.py']
    names += [str(p.relative_to(root)) for directory in ['src', 'include/DPGO']
              for p in (root/directory).glob('Native*')]
    names += ['src/IncrementalLocalModel.cpp', 'include/DPGO/IncrementalLocalModel.h']
    for name in names:
        destination = args.output/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, destination)
        manifest['sources'][name] = sha(root/name)
    def execute(tag, command):
        start = time.perf_counter()
        with (args.output/f'{tag}.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = f'timeout_{args.timeout}s'
        row = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(row)
        dump(args.output/'manifest.json', manifest)
        print(json.dumps(row), flush=True)
        return code == 0
    tag = 'dran'
    command = [str(binaries['dran']), str(args.output/'events.jsonl'), str(args.output/tag),
        args.dran_mode, str(args.duration), '--warm-start', str(args.output/'input.json')]
    command += [f'{a}:{b}' for a, b in package['links']]
    if not execute(tag, command):
        manifest['all_estimators_terminal'] = True
        dump(args.output/'manifest.json', manifest)
        raise SystemExit('DRAN suffix failed; retained diagnostics')
    ref = json.loads((args.output/tag/'summary.json').read_text())
    credits, total = [], 0
    for row in read_rows(args.output/tag/'updates.csv'):
        total += int(row['optimizer_bytes'])+int(row['control_bytes'])
        credits.append(dict(robot=int(row['robot']), timestamp_s=float(row['timestamp_s']), application_bytes=total))
    with (args.output/'credits.jsonl').open('x') as f:
        for row in credits:
            f.write(json.dumps(row)+'\n')
    results = [dict(tag=tag, **verify(args.output/tag, package, ref, credits))]
    for mode in ['imesa_paper', 'rimesa', 'rimesa_gaussian']:
        for schedule in ['one', 'bytes']:
            tag = f'{mode}_{schedule}'
            command = [str(binaries['baseline']), str(args.output/'input.json'), str(args.output/tag),
                mode, '--online', str(args.output/'events.jsonl'), str(args.duration),
                '1' if schedule == 'one' else str(args.output/'credits.jsonl')]
            if execute(tag, command):
                results.append(dict(tag=tag, **verify(args.output/tag, package, ref, credits)))
    manifest['all_estimators_terminal'] = True
    dump(args.output/'manifest.json', manifest)
    if args.raw:
        # Evaluation starts only once every estimator process has terminated.
        from benchmarks.mrclam.evaluate import load_ground_truth
        from benchmarks.mrclam.source import audit_dataset, load_manifest
        from experiments.evaluate_native_joint import evaluate_joint_rows
        from experiments.analyze_joint_delivery import delivery_metrics
        dataset = package['provenance']['dataset']
        data_manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        source = audit_dataset(data_manifest, args.raw)
        truth = load_ground_truth(data_manifest, source.dataset_directory)
        labels = dict(data_manifest.evaluation.object_ground_truth_label_map)
        for row in results:
            folder = args.output/row['tag']
            row['quality'] = evaluate_joint_rows(read_rows(folder/'final_estimates.csv'), truth, labels)
            causal = read_rows(folder/'causal_estimates.csv')
            row['causal_quality'] = evaluate_joint_rows(causal, truth, labels)
            row['delivery'] = delivery_metrics(causal, read_rows(folder/'updates.csv'), truth, row['causal_quality'])
    dump(args.output/'comparison.json', dict(results=results,
        warning='Infeasible copies can lower the physical loss. Fixed opportunity and matched-byte arms are not equal time. Common past-state generation excluded for all. Single serial timing realization, zero network latency.'))
    with (args.output/'summary.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'bytes', 'physical_cost', 'consensus_m', 'setup_s',
            'service_s', 'backlog_s', 'trajectory_rmse_m', 'object_rmse_m', 'causal_rmse_m', 'delivery_rmse_m'])
        writer.writeheader()
        for row in results:
            writer.writerow(dict(tag=row['tag'], bytes=row['application_bytes'], physical_cost=row['physical_cost'],
                consensus_m=row['consensus_max_m'], setup_s=row['setup_seconds'], service_s=row['service_seconds'],
                backlog_s=row['serial_max_backlog_s'], trajectory_rmse_m=row.get('quality', {}).get('translation_rmse_m'),
                object_rmse_m=row.get('quality', {}).get('object_copy_rmse_m'),
                causal_rmse_m=row.get('causal_quality', {}).get('translation_rmse_m'),
                delivery_rmse_m=row.get('delivery', {}).get('delivery_time_stale_pose_rmse_m')))
    if any(r['exit_code'] != 0 for r in manifest['runs']):
        raise SystemExit('Some runs failed; see retained manifest/logs')


if __name__ == '__main__':
    main()
