"""Fixed added-variance attribution audit; not an adaptive estimator."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_conditional_replay import compare_runs, read_rows, sha256
from experiments.audit_lazy_refresh import audit_updates, dump
from experiments.analyze_full_lazy_refresh import recovery_diagnostics


def arms():
    return [('nominal1', 0.), ('added1', .0225), ('added2', .0225), ('nominal2', 0.)]


def cases():
    return [('clean_lazy', 'd2_clean', True), ('gaussian_lazy', 'd2_gaussian', True),
            ('gaussian_ordinary', 'd2_gaussian', False)]


def verify_problem(reference, result):
    for key in ['frames', 'packets', 'physical_factors', 'prefix_physical_factors', 'initial_cost']:
        if reference[key] != result[key]:
            raise ValueError(f'{key} information/prefix mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('/tmp/dranppr_backend_build/bin/native-joint-replay'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prior = root/'results/dranppr_lazy_transfer_stress_20261003'
    stress = json.loads((prior/'inputs/gaussian_bearing/stress_manifest.json').read_text())
    if (stress['kind'] != 'gaussian_bearing' or stress['fixed_condition']['additive_bearing_sd_rad'] != .15
            or not stress['prefix_byte_identical']):
        raise ValueError('unexpected injection contract')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'native-joint-replay'
    shutil.copy2(args.binary, binary)
    shutil.copy2(prior/'d2_clean/warm_start.json', output/'warm_start.json')
    if sha256(output/'warm_start.json') != stress['warm_start_sha256']:
        raise ValueError('stress/common-past mismatch')
    package = json.loads((output/'warm_start.json').read_text())
    if package.get('truth_used', False) or package['links'] != [[1,2],[2,3],[3,4],[4,5]]:
        raise ValueError('unexpected warm-start contract')
    inputs = {}
    for name in ['d2_clean', 'd2_gaussian']:
        destination = output/(name+'_events.jsonl')
        shutil.copy2(prior/name/'events.jsonl', destination)
        inputs[name] = dict(path=str(destination), sha256=sha256(destination))
    if inputs['d2_gaussian']['sha256'] != stress['events_sha256']:
        raise ValueError('Gaussian stream provenance mismatch')
    if inputs['d2_clean']['sha256'] != stress['original_events_sha256']:
        raise ValueError('original stream provenance mismatch')
    shutil.copy2(prior/'inputs/gaussian_bearing/stress_manifest.json', output/'stress_manifest.json')
    shutil.copy2('/tmp/dranppr_backend_build/CMakeCache.txt', output/'CMakeCache.txt')
    names = ['CMakeLists.txt', 'cmake/native_local_model.cmake', 'experiments/online_stream.h',
        'experiments/audit_oracle_variance.py', 'experiments/audit_lazy_refresh.py',
        'experiments/audit_conditional_replay.py', 'experiments/analyze_full_lazy_refresh.py',
        'experiments/evaluate_native_joint.py', 'experiments/analyze_joint_delivery.py',
        'tests/test_native_suffix_noise.py', 'tests/test_audit_oracle_variance.py',
        'tests/test_native_conditional_replay.py', 'tests/test_native_joint_audit.py',
        'tests/test_native_matched_online.py']
    sources = [root/name for name in names]+list((root/'experiments/native_local').glob('*'))
    for directory, suffix in [('src', '.cpp'), ('include/DPGO', '.h')]:
        for stem in ['Native*', 'Incremental*']:
            sources += list((root/directory).glob(stem+suffix))
    for source in sources:
        if source.is_file():
            destination = output/'source'/source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(runs=[], inputs=inputs, binary_sha256=sha256(binary),
        warm_start_sha256=sha256(output/'warm_start.json'),
        libraries=subprocess.check_output(['ldd', str(binary)], text=True),
        env={k:env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
        duration_s=120., interface_iterations_per_frame=1, timeout_s=600,
        all_estimators_terminal=False, truth_state_used_by_estimator=False,
        noise_learned=False, noise_information='Added variance is externally supplied. Oracle for injected Gaussian component only; deliberate overinflation on original data.',
        comparison='Same causal observations and iteration cap, not equal bytes/compute. Own weighted losses are NOT comparable across variance settings.')
    results, parity = {}, {}
    for case, input_name, lazy in cases():
        results[case] = {}
        for tag, variance in arms():
            folder = output/case/tag
            folder.parent.mkdir(exist_ok=True)
            command = [str(binary), inputs[input_name]['path'], str(folder), 'joint_cached_one', '120',
                '--warm-start', str(output/'warm_start.json'), '--suffix-added-bearing-variance', str(variance)]
            if lazy:
                command.append('--lazy-refresh')
            command += ['1:2', '2:3', '3:4', '4:5']
            start = time.perf_counter()
            with (output/case/(tag+'.log')).open('x') as log:
                try:
                    code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=600).returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout_600s'
            record = dict(case=case, tag=tag, command=command, exit_code=code,
                          wall_seconds=time.perf_counter()-start)
            manifest['runs'].append(record)
            dump(output/'manifest.json', manifest)
            print(json.dumps(record), flush=True)
            if code != 0:
                raise RuntimeError('Replay failed; retained output, stopping experiment')
            summary = json.loads((folder/'summary.json').read_text())
            if (summary['suffix_added_bearing_variance'] != variance or
                    not math.isclose(summary['suffix_bearing_sigma'], math.sqrt(.05**2+variance)) or
                    summary['noise_parameters_learned']):
                raise ValueError('noise model configuration mismatch')
            rows = read_rows(folder/'updates.csv')
            checks = audit_updates(rows, lazy, summary['refresh_pending'])
            if summary['root_displacement'] != 0. or summary['consensus_max_m'] > 1e-10:
                raise ValueError('gauge/consensus mismatch')
            for key in ['application_bytes', 'refresh_requests', 'refresh_rounds']:
                if checks[key] != summary[key]:
                    raise ValueError(f'{key} ledger mismatch')
            control_total = 0
            for line in (folder/'control_messages.jsonl').open():
                message = json.loads(line)
                if abs(message['from']-message['to']) != 1:
                    raise ValueError('control message outside topology')
                control_total += message['application_bytes']
            if control_total != summary['control_application_bytes']:
                raise ValueError('control byte ledger mismatch')
            if results[case]:
                verify_problem(results[case]['nominal1'], summary)
                identity = lambda r: (r['robot'], r['timestamp_s'])
                if list(map(identity, rows)) != list(map(identity, read_rows(output/case/'nominal1/updates.csv'))):
                    raise ValueError('different causal frame identities')
            results[case][tag] = dict(summary, audit=checks, recovery=recovery_diagnostics(rows))
        pairs = [('nominal_repeat', output/case/'nominal1', output/case/'nominal2'),
                 ('added_repeat', output/case/'added1', output/case/'added2'),
                 ('previous_default', prior/input_name/('candidate1' if lazy else 'reference1'),
                  output/case/'nominal1')]
        parity[case] = {}
        for name, left, right in pairs:
            comparison = compare_runs(left, right)
            for filename in ['final_estimates.csv', 'causal_estimates.csv', 'control_messages.jsonl']:
                if sha256(left/filename) != sha256(right/filename):
                    raise ValueError(f'{name}: {filename} changed')
            if comparison['physical_cost_abs'] != 0. or comparison['application_bytes_difference'] != 0:
                raise ValueError('default/repeat numerical or traffic mismatch')
            parity[case][name] = comparison
    manifest['all_estimators_terminal'] = True
    dump(output/'manifest.json', manifest)
    # Oracle variance comes from the injection model; GT states are posthoc only.
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    from experiments.analyze_joint_delivery import delivery_metrics
    data_manifest = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    data_audit = audit_dataset(data_manifest, args.raw)
    truth = load_ground_truth(data_manifest, data_audit.dataset_directory)
    labels = dict(data_manifest.evaluation.object_ground_truth_label_map)
    for case, runs in results.items():
        for tag, result in runs.items():
            folder = output/case/tag
            result['quality'] = evaluate_joint_rows(read_rows(folder/'final_estimates.csv'), truth, labels)
            causal = read_rows(folder/'causal_estimates.csv')
            result['causal_quality'] = evaluate_joint_rows(causal, truth, labels)
            result['delivery'] = delivery_metrics(causal, read_rows(folder/'updates.csv'), truth, result['causal_quality'])
    dump(output/'analysis.json', dict(results=results, implementation_parity=parity,
        warning=manifest['comparison'], truth_used_by_evaluation=True))
    for case, runs in results.items():
        print(json.dumps(dict(case=case, runs={tag:dict(service_s=r['service_seconds'],
            bytes=r['application_bytes'], trajectory_rmse_m=r['quality']['translation_rmse_m'],
            object_rmse_m=r['quality']['object_copy_rmse_m'],
            causal_rmse_m=r['causal_quality']['translation_rmse_m']) for tag,r in runs.items()})), flush=True)


if __name__ == '__main__':
    main()
