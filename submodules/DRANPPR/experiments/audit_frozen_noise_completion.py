"""Predeclared covariance-preserving offline completion, not noise adaptation."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_conditional_replay import read_rows, sha256
from experiments.audit_lazy_refresh import dump
from experiments.prepare_frozen_completion import prepare
from experiments.prepare_lazy_stress import perturb


def grid():
    result = []
    for case in ['original', 'gaussian_20261003', 'gaussian_20261004', 'gaussian_20261005']:
        for objective in ['nominal', 'added']:
            for start in ['nominal', 'added']:
                result.append((case, objective, start, 'central_lm'))
            result.append((case, objective, objective, 'central_gn'))
            if case == 'gaussian_20261003':
                result.append((case, objective, objective, 'routed'))
    return result


def state_difference(left, right):
    key = lambda r: (int(r['robot']), r['kind'], int(r['index']))
    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    if set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError('state identity mismatch')
    xy, yaw = [], []
    for k, p in a.items():
        q = b[k]
        xy.append(math.hypot(float(p['x'])-float(q['x']), float(p['y'])-float(q['y'])))
        if k[1] == 'pose':
            d = float(p['yaw'])-float(q['yaw'])
            yaw.append(abs(math.atan2(math.sin(d), math.cos(d))))
    return dict(max_xy_m=max(xy), rms_xy_m=math.sqrt(sum(d*d for d in xy)/len(xy)),
                max_yaw_rad=max(yaw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prior = root/'results/dranppr_oracle_variance_20261003'
    previous = json.loads((prior/'manifest.json').read_text())
    if not previous['all_estimators_terminal']:
        raise ValueError('prior estimators incomplete')
    for item in previous['inputs'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('prior stream hash mismatch')
    if sha256(prior/'warm_start.json') != previous['warm_start_sha256']:
        raise ValueError('prior prefix hash mismatch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(prior/'warm_start.json', output/'warm_start.json')
    shutil.copy2(prior/'manifest.json', output/'prior_manifest.json')
    warm = json.loads((output/'warm_start.json').read_text())
    first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    inputs = {}
    for name, filename in [('original', 'd2_clean_events.jsonl'), ('gaussian_20261003', 'd2_gaussian_events.jsonl')]:
        path = output/(name+'.jsonl')
        shutil.copy2(prior/filename, path)
        inputs[name] = path
    for seed in [20261004, 20261005]:
        name = 'gaussian_'+str(seed)
        lines, stress = perturb(inputs['original'].read_text().splitlines(keepends=True),
                                cutoff, first+120., 'gaussian_bearing', seed)
        inputs[name] = output/(name+'.jsonl')
        inputs[name].write_text(''.join(lines))
        stress.update(original_events_sha256=sha256(inputs['original']),
                      events_sha256=sha256(inputs[name]), warm_start_sha256=sha256(output/'warm_start.json'))
        dump(output/(name+'_stress.json'), stress)
    shutil.copy2(prior/'stress_manifest.json', output/'gaussian_20261003_stress.json')
    binaries = {}
    for name in ['native-joint-audit', 'native-joint-replay']:
        binaries[name] = output/name
        shutil.copy2(Path('/tmp/dranppr_backend_build/bin')/name, binaries[name])
    names = ['CMakeLists.txt', 'cmake/native_local_model.cmake', 'experiments/online_stream.h',
             'experiments/audit_frozen_noise_completion.py', 'experiments/prepare_frozen_completion.py',
             'experiments/prepare_lazy_stress.py', 'experiments/evaluate_native_joint.py',
             'experiments/analyze_oracle_variance.py', 'tests/test_frozen_noise_completion.py',
             'docs/superpowers/plans/2026-10-03-frozen-noise-completion.md']
    files = [root/n for n in names]+list((root/'experiments/native_local').glob('*'))
    for directory, suffix in [('src', '.cpp'), ('include/DPGO', '.h')]:
        for stem in ['Native*', 'Incremental*']:
            files += list((root/directory).glob(stem+suffix))
    for source in files:
        if source.is_file():
            target = output/'source'/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2('/tmp/dranppr_backend_build/CMakeCache.txt', output/'CMakeCache.txt')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(inputs={k:dict(path=str(v), sha256=sha256(v)) for k,v in inputs.items()},
        binaries={k:sha256(v) for k,v in binaries.items()}, warm_start_sha256=sha256(output/'warm_start.json'),
        libraries=subprocess.check_output(['ldd', str(binaries['native-joint-audit'])], text=True),
        env={k:env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}, grid=grid(),
        runs=[], reused_online={}, all_estimators_terminal=False, truth_used_by_estimator=False,
        noise_learned=False, duration_s=120., max_iterations=100, timeout_s=600,
        warning='Offline completion; own losses only comparable within a covariance objective. Single D2 sequence, fixed noise seeds; no speed or global optimality claim.')
    dump(output/'manifest.json', manifest)

    def execute(command, tag):
        start = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_600s'
        record = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(record)
        dump(output/'manifest.json', manifest)
        print(json.dumps(record), flush=True)
        if code != 0:
            raise RuntimeError('Completion failed; retained evidence, no parameter retries')

    online, events = {}, {}
    for case, path in inputs.items():
        events[case] = [json.loads(line) for line in path.open()]
        online[case] = {}
        for objective, variance in [('nominal', 0.), ('added', .0225)]:
            folder = output/'online'/case/objective
            folder.parent.mkdir(parents=True, exist_ok=True)
            if case in ['original', 'gaussian_20261003']:
                oldcase = 'clean_lazy' if case == 'original' else 'gaussian_lazy'
                source = prior/oldcase/(objective+'1')
                shutil.copytree(source, folder)
                manifest['reused_online'][case+'/'+objective] = dict(source=str(source),
                    hashes={f:sha256(source/f) for f in ['summary.json', 'final_estimates.csv', 'causal_estimates.csv']})
            else:
                command = [str(binaries['native-joint-replay']), str(path), str(folder),
                    'joint_cached_one', '120', '--warm-start', str(output/'warm_start.json'),
                    '--suffix-added-bearing-variance', str(variance), '--lazy-refresh', '1:2', '2:3', '3:4', '4:5']
                execute(command, 'online_'+case+'_'+objective)
            summary = json.loads((folder/'summary.json').read_text())
            if (summary['physical_factors'] != 3538 or summary['frames'] != 1222 or
                    summary['suffix_added_bearing_variance'] != variance or summary['root_displacement'] != 0. or
                    summary['consensus_max_m'] != 0.):
                raise ValueError('online graph/gauge/noise contract mismatch')
            online[case][objective] = dict(folder=folder, summary=summary)
    completions = []
    for case, objective, start, mode in grid():
        tag = '_'.join([case, objective, 'from', start, mode])
        initial = online[case][start]
        package = prepare(events[case], read_rows(initial['folder']/'final_estimates.csv'), warm, 120.,
                          .0225 if objective == 'added' else 0.)
        package_path = output/(tag+'.json')
        dump(package_path, package)
        folder = output/'frozen'/tag
        execute([str(binaries['native-joint-audit']), str(package_path), str(folder), mode, '100'], tag)
        summary = json.loads((folder/'summary.json').read_text())
        if start == objective and not math.isclose(summary['initial_cost'], initial['summary']['physical_cost'],
                                                  rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('export changed matched online physical cost')
        completions.append(dict(case=case, objective=objective, start=start, mode=mode, tag=tag,
                                folder=str(folder), summary=summary, package_sha256=sha256(package_path)))
    manifest['all_estimators_terminal'] = True
    dump(output/'manifest.json', manifest)
    # No ground-truth loading until every estimator subprocess has terminated.
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows, package_rows
    from experiments.analyze_oracle_variance import point_error_squared
    data = load_manifest(root/'benchmarks/mrclam/manifests/dataset2.json')
    source = audit_dataset(data, args.raw)
    truth = load_ground_truth(data, source.dataset_directory)
    labels = dict(data.evaluation.object_ground_truth_label_map)
    alignment = evaluate_joint_rows(package_rows(warm), truth, labels)
    analysis = dict(completions=completions, online={}, comparisons=[], common_prefix_alignment=alignment,
                    warning=manifest['warning'], truth_used_by_evaluation=True)
    for case, arms in online.items():
        analysis['online'][case] = {}
        for objective, item in arms.items():
            causal = read_rows(item['folder']/'causal_estimates.csv')
            errors = [point_error_squared(row, truth.robots[int(row['robot'])].interpolate(float(row['timestamp_s']))[0], alignment)
                      for row in causal if truth.robots[int(row['robot'])].supports(float(row['timestamp_s']))]
            analysis['online'][case][objective] = dict(summary=item['summary'],
                quality=evaluate_joint_rows(read_rows(item['folder']/'final_estimates.csv'), truth, labels),
                causal_quality=evaluate_joint_rows(causal, truth, labels),
                common_alignment_causal_rmse_m=math.sqrt(sum(errors)/len(errors)))
    for item in completions:
        rows = read_rows(Path(item['folder'])/'final_estimates.csv')
        item['quality'] = evaluate_joint_rows(rows, truth, labels)
        reference = next(r for r in completions if r['case'] == item['case'] and
                         r['objective'] == item['objective'] and r['start'] == item['objective'] and r['mode'] == 'central_lm')
        analysis['comparisons'].append(dict(tag=item['tag'], reference=reference['tag'],
            cost_difference=item['summary']['final_cost']-reference['summary']['final_cost'],
            **state_difference(rows, read_rows(Path(reference['folder'])/'final_estimates.csv'))))
    dump(output/'analysis.json', analysis)
    print(json.dumps(dict(completed=len(completions), analysis=str(output/'analysis.json'))), flush=True)


if __name__ == '__main__':
    main()
