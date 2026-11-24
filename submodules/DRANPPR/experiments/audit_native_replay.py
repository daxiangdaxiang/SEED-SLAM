"""Serial causal Gaussian controls and cost attribution; truth is posthoc only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from evaluate_incremental_reference import evaluate_rows, read_rows
from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def profile(rows):
    if not rows:
        raise ValueError('empty frame profile')
    milliseconds = np.array([float(r['total_ms'])+float(r['initialization_ms']) for r in rows])
    components = {key: sum(float(r[key]) for r in rows)/1000.
                  for key in rows[0] if key.endswith('_ms')}
    refreshed = sum(int(r.get('model_refresh_factors', 0)) for r in rows)
    reused = sum(int(r.get('model_refresh_reused', 0)) for r in rows)
    summary = dict(
        frames=len(rows), measured_work_s=float(milliseconds.sum()/1000.),
        update_p50_ms=float(np.quantile(milliseconds, .5)),
        update_p95_ms=float(np.quantile(milliseconds, .95)),
        update_p99_ms=float(np.quantile(milliseconds, .99)),
        update_max_ms=float(milliseconds.max()),
        max_serial_backlog_s=max(float(r['serial_backlog_s']) for r in rows),
        max_independent_robot_backlog_s=max(float(r['robot_backlog_s']) for r in rows),
        accepted_steps=sum(int(r['accepted_steps']) for r in rows),
        model_refresh_factors=refreshed, model_refresh_reused=reused,
        model_refresh_fraction=refreshed/(refreshed+reused) if refreshed+reused else None,
        descent_violations=sum(float(r['cost_after']) > float(r['cost_before'])+1e-10
                               for r in rows if r.get('cost_before')),
        components_s=components,
        timing_note='Components contain nested counters. Total is ONLY total_ms + initialization_ms; do not sum all components.')
    growth = []
    for robot in sorted({int(r['robot']) for r in rows}):
        local = [r for r in rows if int(r['robot']) == robot]
        for index, group in enumerate(np.array_split(np.arange(len(local)), 4)):
            if not len(group):
                continue
            subset = [local[int(i)] for i in group]
            ms = np.array([float(r['total_ms']) for r in subset])
            growth.append(dict(robot=robot, frame_quartile=index+1, frames=len(subset),
                elapsed_start_s=float(subset[0]['elapsed_s']), elapsed_end_s=float(subset[-1]['elapsed_s']),
                max_history_values=max(int(r['history_values']) for r in subset),
                service_mean_ms=float(ms.mean()), service_p99_ms=float(np.quantile(ms, .99)),
                max_robot_backlog_s=max(float(r['robot_backlog_s']) for r in subset)))
    return summary, growth


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--gtsam-source', type=Path, required=True)
    parser.add_argument('--gtsam-library', type=Path, required=True)
    parser.add_argument('--stream-root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', type=int, nargs='+', default=[1, 4, 9])
    parser.add_argument('--modes', nargs='+', choices=['native', 'native_cached', 'isam2_exact'],
                        default=['native', 'native_cached', 'isam2_exact'])
    parser.add_argument('--duration', type=float, default=120.)
    parser.add_argument('--timeout', type=float, default=300.)
    args = parser.parse_args()
    if args.duration <= 30 or args.timeout <= 0:
        parser.error('require duration > 30 and positive timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = args.output / 'source_snapshot'
    source.mkdir()
    paths = [Path(__file__), root/'experiments/native_local/replay.cpp',
             root/'experiments/evaluate_incremental_reference.py', root/'cmake/native_local_model.cmake',
             root/'tests/test_native_causal_replay.py', root/'tests/test_native_replay_profile.py']
    for name in ['IncrementalLocalModel', 'NativeSchurModel', 'NativeLocalOptimizer', 'NativeBetweenPose2Factor']:
        paths += [root/f'include/DPGO/{name}.h', root/f'src/{name}.cpp']
    for path in paths:
        shutil.copy2(path, source/path.name)
    def git(*command):
        return subprocess.check_output(['git', '-C', str(args.gtsam_source), *command], text=True).strip()
    (source/'gtsam.patch').write_text(git('diff', '--no-ext-diff')+'\n')
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        binary_sha256=sha256(args.binary), gtsam_library_sha256=sha256(args.gtsam_library),
        gtsam_commit=git('rev-parse', 'HEAD'), gtsam_dirty=git('status', '--porcelain'),
        dynamic_libraries=subprocess.check_output(['ldd', str(args.binary)], text=True),
        source_hashes={p.name: sha256(p) for p in source.iterdir()}, streams={},
        likelihood='Identical exact-Jacobian Pose2 odometry sigma=(.1,.1,.05)*sqrt(dt); native polar sigma=(.05,.1)',
        identity='declared barcode identities, no association edits',
        initialization='same rule, current own estimate + arrived odometry; own first polar sighting for landmark; no GT',
        methods='native: one full GN/Schur step with up to 30 true-cost backtracks; native_cached: same step from declared bases, refresh=1e-4; iSAM2: QR, relinearize=1e-4, skip=1, wildfire=1e-3',
        scope='independent local Gaussian models, not robust/distributed; bounded iterations is not bounded work',
        evaluation='independent per-robot posthoc SE2 trajectory alignment, including causal outputs; not an online gauge fix',
        timing='serial mode runs; full process wall includes parsing/CSV; service excludes these and posthoc GT; nested timers not additive')
    manifest_path = args.output/'manifest.json'
    manifest_path.write_text(json.dumps(metadata, indent=2)+'\n')
    completed, failures = [], []
    for dataset in args.datasets:
        stream = args.stream_root/f'dataset{dataset}'
        stream_meta = json.loads((stream/'metadata.json').read_text())
        if sha256(stream/'events.jsonl') != stream_meta['event_sha256']:
            raise ValueError(f'input hash mismatch: {stream}')
        metadata['streams'][dataset] = stream_meta
        manifest_path.write_text(json.dumps(metadata, indent=2)+'\n')
        for mode in args.modes:
            tag = f'dataset{dataset}_{mode}'
            output = args.output/tag
            command = [str(args.binary), str(stream/'events.jsonl'), str(output), mode, str(args.duration)]
            start = time.monotonic()
            try:
                with (args.output/f'{tag}.log').open('w') as log:
                    run = subprocess.run(command, stdout=log, stderr=log, timeout=args.timeout)
                wall = time.monotonic()-start
                if run.returncode:
                    raise RuntimeError(f'exit {run.returncode}; see {tag}.log')
                completed.append((dataset, mode, output, wall))
                print(f'{tag}: completed, wall={wall:.3f}s', flush=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append(dict(dataset=dataset, mode=mode, command=command, error=str(error),
                                     wall_s=time.monotonic()-start))
                print(f'{tag}: FAILED {error}', flush=True)
            (args.output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    # All estimator subprocesses have exited before any truth is loaded.
    summaries, robot_rows, growth_rows, profiles = [], [], [], {}
    truths = {}
    for dataset,mode,output,wall in completed:
        if dataset not in truths:
            manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
            audit = audit_dataset(manifest, args.raw)
            truths[dataset] = (load_ground_truth(manifest, audit.dataset_directory),
                               dict(manifest.evaluation.object_ground_truth_label_map))
        truth, label_map = truths[dataset]
        run = json.loads((output/'summary.json').read_text())
        updates = read_rows(output/'updates.csv')
        stats, growth = profile(updates)
        if abs(stats['measured_work_s']-run['total_work_s']) > 1e-8:
            raise ValueError('service ledger mismatch')
        if stats['descent_violations']:
            raise ValueError('native current-prefix physical objective increased')
        profiles[f'dataset{dataset}_{mode}'] = stats
        growth_rows.extend(dict(dataset=dataset, mode=mode, **row) for row in growth)
        first = min(float(row['timestamp_s']) for row in updates)
        for period in ['all', 'after30']:
            for state in ['final', 'causal']:
                rows = read_rows(output/f'{state}_estimates.csv')
                if period == 'after30':
                    rows = [row for row in rows if row['kind'] == 'landmark' or float(row['timestamp_s'])-first > 30.]
                aggregate, robots = evaluate_rows(rows, truth, label_map)
                summaries.append(dict(dataset=dataset, mode=mode, period=period, state=state, **aggregate,
                    **{k:v for k,v in stats.items() if k not in ('components_s', 'timing_note')},
                    wall_s=wall, physical_gaussian_cost=run['physical_gaussian_cost']))
                robot_rows.extend(dict(dataset=dataset, mode=mode, period=period, state=state, **row) for row in robots)
    write_csv(args.output/'summary.csv', summaries)
    write_csv(args.output/'robots.csv', robot_rows)
    write_csv(args.output/'growth.csv', growth_rows)
    (args.output/'profiles.json').write_text(json.dumps(profiles, indent=2)+'\n')
    if failures:
        raise SystemExit(f'{len(failures)} failures retained; report them with successful cases')


if __name__ == '__main__':
    main()
