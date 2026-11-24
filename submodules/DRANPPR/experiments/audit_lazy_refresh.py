"""Bounded, common-past test of rejection-triggered historical model refresh."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from experiments.audit_conditional_replay import compare_runs, read_rows, sha256


def audit_updates(rows, lazy, pending):
    previous = 0
    total = requests = rounds = failures = accepted = span = longest = 0
    ratios = []
    for row in rows:
        before, after = float(row['cost_before']), float(row['cost_after'])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError('nonfinite physical cost')
        if after > before+1e-10:
            raise ValueError('physical descent violated')
        if int(row['pcg_iterations']) > 1:
            raise ValueError('one-step interface iteration cap violated')
        request, refresh = int(row['refresh_requests']), int(row['refresh_rounds'])
        if lazy and refresh != previous:
            raise ValueError('deferred refresh not performed at next frame')
        prediction = json.loads(row['full_step_prediction'])
        decrease = json.loads(row['full_step_decrease'])
        failed = prediction is None or prediction <= 0 or decrease is None or decrease <= 0
        if request != int(lazy and failed):
            raise ValueError('full-step decision mismatch')
        if not lazy and refresh:
            raise ValueError('baseline activated lazy refresh')
        if prediction is not None and prediction > 0 and decrease is not None:
            ratios.append(decrease/prediction)
        failures += failed
        requests += request
        rounds += refresh
        accepted += int(row['accepted'])
        span = 0 if refresh else span+1
        longest = max(longest, span)
        total += int(row['optimizer_bytes'])+int(row['control_bytes'])
        previous = request
    if pending != bool(previous):
        raise ValueError('terminal pending refresh not reported correctly')
    return dict(application_bytes=total, refresh_requests=requests, refresh_rounds=rounds,
                full_step_failures=failures, accepted_frames=accepted,
                longest_no_refresh_frames=longest,
                full_step_gain_ratio_min=min(ratios) if ratios else None,
                full_step_gain_ratio_median=statistics.median(ratios) if ratios else None)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replay_arms(normal_only):
    arms = [('reference1', False, False), ('candidate1', not normal_only, True),
            ('candidate2', not normal_only, True), ('reference2', False, False)]
    if not normal_only:
        arms += [('conditional_reference', True, False), ('lazy_normal', False, True)]
    return arms


def select_inputs(root, dataset, events, warm):
    if (events is None) != (warm is None):
        raise ValueError('events and warm-start must be supplied together')
    if events is not None:
        return events, warm
    if dataset not in (4, 9):
        raise ValueError('explicit events and warm-start required for transfer datasets')
    source = root/'results/dranppr_nonlinear_conditional_20261002'
    return source/f'inputs/d{dataset}_events.jsonl', source/f'd{dataset}_one/warm_start.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('/tmp/dranppr_backend_build/bin/native-joint-replay'))
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4, 9], required=True)
    parser.add_argument('--events', type=Path)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--legacy-reference', type=Path)
    parser.add_argument('--normal-only', action='store_true', help='Normal versus normal lazy ABBA, without conditional controls')
    parser.add_argument('--timeout', type=float, default=240., help='Per-process watchdog, not an optimizer budget')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    source_events, source_warm = select_inputs(root, args.dataset, args.events, args.warm_start)
    package = json.loads(source_warm.read_text())
    if package.get('truth_used', False) or package['provenance']['dataset'] != args.dataset:
        raise ValueError('warm-start dataset or truth provenance mismatch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'native-joint-replay'
    events, warm = output/'events.jsonl', output/'warm_start.json'
    shutil.copy2(args.binary, binary)
    shutil.copy2(source_events, events)
    shutil.copy2(source_warm, warm)
    sources = [root/'CMakeLists.txt', root/'cmake/native_local_model.cmake',
               root/'experiments/online_stream.h', Path(__file__),
               root/'experiments/audit_conditional_replay.py',
               root/'experiments/evaluate_native_joint.py', root/'experiments/analyze_joint_delivery.py']
    sources += list((root/'experiments/native_local').glob('*'))
    for stem in ['Native*', 'Incremental*']:
        sources += list((root/'include/DPGO').glob(stem+'.h'))
        sources += list((root/'src').glob(stem+'.cpp'))
    sources += list((root/'tests').glob('*[Ll]azy[Rr]efresh*'))
    sources += [root/'tests/test_native_lazy_refresh.py', root/'tests/test_audit_lazy_refresh.py']
    for path in sources:
        if path.is_file():
            target = output/'source'/path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2('/tmp/dranppr_backend_build/CMakeCache.txt', output/'CMakeCache.txt')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(dataset=args.dataset, duration=args.duration, runs=[],
        source_events=str(source_events), source_warm_start=str(source_warm),
        binary_sha256=sha256(binary), events_sha256=sha256(events), warm_start_sha256=sha256(warm),
        libraries=subprocess.check_output(['ldd', str(binary)], text=True),
        env={k:env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
        normal_only=args.normal_only, timeout_seconds=args.timeout,
        scope=('One interface iteration/frame. Normal versus normal lazy ABBA.' if args.normal_only else
               'One interface iteration/frame. ABBA service comparison plus single-run algebra/policy controls.')+
              ' Not matched bytes, independent seeds, or asynchronous deployment.',
        truth_used_by_estimator=False, all_estimators_terminal=False)
    arms = replay_arms(args.normal_only)
    results = {}
    for tag, conditional, lazy in arms:
        mode = 'joint_conditional_cached_one' if conditional else 'joint_cached_one'
        command = [str(binary), str(events), str(output/tag), mode, str(args.duration), '--warm-start', str(warm)]
        if lazy:
            command += ['--lazy-refresh']
        command += ['1:2', '2:3', '3:4', '4:5']
        start = time.perf_counter()
        with (output/f'{tag}.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = f'timeout_{args.timeout:g}s'
        record = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(record)
        dump(output/'manifest.json', manifest)
        print(json.dumps(record), flush=True)
        if code != 0:
            raise RuntimeError('Replay failed; logs retained, no additional experiments launched')
        summary = json.loads((output/tag/'summary.json').read_text())
        checks = audit_updates(read_rows(output/tag/'updates.csv'), lazy, summary['refresh_pending'])
        if summary['root_displacement'] != 0 or summary['consensus_max_m'] > 1e-10:
            raise ValueError('gauge/consensus failure')
        for key in ['application_bytes', 'refresh_requests', 'refresh_rounds']:
            if checks[key] != summary[key]:
                raise ValueError(f'{key} ledger mismatch')
        total = 0
        for line in (output/tag/'control_messages.jsonl').open():
            row = json.loads(line)
            if abs(row['from']-row['to']) != 1:
                raise ValueError('control message outside chain')
            total += row['application_bytes']
        if total != summary['control_application_bytes']:
            raise ValueError('control ledger mismatch')
        if results:
            reference = results['reference1']
            for key in ['frames', 'packets', 'physical_factors', 'prefix_physical_factors', 'initial_cost']:
                if reference[key] != summary[key]:
                    raise ValueError(f'{key} input mismatch')
        results[tag] = dict(summary, audit=checks)
    manifest['all_estimators_terminal'] = True
    dump(output/'manifest.json', manifest)
    if args.normal_only:
        parity = dict(reference_repeat=compare_runs(output/'reference1', output/'reference2'),
                      candidate_repeat=compare_runs(output/'candidate1', output/'candidate2'))
    else:
        parity = dict(ordinary=compare_runs(output/'reference1', output/'conditional_reference'),
                      lazy=compare_runs(output/'lazy_normal', output/'candidate1'))
    if args.legacy_reference:
        parity['legacy'] = compare_runs(args.legacy_reference, output/'reference1')
    for key, comparison in parity.items():
        if comparison['final_state_max_abs'] > 1e-7 or comparison['causal_state_max_abs'] > 1e-7:
            raise ValueError(f'{key} implementation parity failure')
    # Ground truth is loaded only after every estimator has terminated.
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    from experiments.analyze_joint_delivery import delivery_metrics
    data_manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{args.dataset}.json')
    data_audit = audit_dataset(data_manifest, args.raw)
    truth = load_ground_truth(data_manifest, data_audit.dataset_directory)
    labels = dict(data_manifest.evaluation.object_ground_truth_label_map)
    for tag, result in results.items():
        result['quality'] = evaluate_joint_rows(read_rows(output/tag/'final_estimates.csv'), truth, labels)
        causal = read_rows(output/tag/'causal_estimates.csv')
        result['causal_quality'] = evaluate_joint_rows(causal, truth, labels)
        result['delivery'] = delivery_metrics(causal, read_rows(output/tag/'updates.csv'), truth, result['causal_quality'])
    dump(output/'analysis.json', dict(results=results, implementation_parity=parity,
        truth_used_by_evaluation=True,
        warning='Known identities, unchanged Gaussian estimator noise model. Explicit input streams may contain artificial perturbations; consult their stress manifests. No robust-outlier, unknown noise, or asynchronous deployment claim. Repeated runtimes are not independent statistical samples.'))
    print(json.dumps({tag:{k:r[k] for k in ['physical_cost', 'service_seconds', 'application_bytes',
        'refresh_requests', 'conditionals_reused', 'quality']} for tag,r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
