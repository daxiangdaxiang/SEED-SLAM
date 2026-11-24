"""Frozen D1/D4 noise-model transfer and past-only block prediction diagnostic."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.audit_native_checkpoint import sha256, typed_state
from experiments.audit_lazy_refresh import dump
from experiments.audit_frozen_noise_completion import state_difference
from experiments.evaluate_incremental_reference import read_rows
from experiments.prepare_frozen_completion import prepare
from experiments.network_noise_support import apply_scales
from experiments.network_noise_prediction import sparse_model, forecast

ARMS = ('nominal', 'joint_r', 'joint_qr', 'split_qr')
CHECKPOINTS = (30, 60, 90, 120)
LINKS = ['1:2', '2:3', '3:4', '4:5']


def grid():
    return [(d,t,a) for d in (1,4) for t in CHECKPOINTS for a in ARMS]


def prediction_summary(rows):
    if not rows:
        raise ValueError('empty predictive support')
    return dict(count=len(rows), mean_nll=float(np.mean([r['nll'] for r in rows])),
        mean_energy=float(np.mean([r['energy'] for r in rows])),
        coverage_95=float(np.mean([r['energy'] <= 5.991464547107979 for r in rows])))


def run(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if (output/'manifest.json').exists():
        raise ValueError('refuse to overwrite existing experiment')
    output.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for name in ['native-joint-audit', 'native-joint-replay']:
        binaries[name] = output/name
        shutil.copy2(Path('/tmp/dranppr_backend_build/bin')/name, binaries[name])
    names = ['experiments/audit_network_noise_transfer.py', 'experiments/network_noise_prediction.py',
        'experiments/audit_network_noise_support.py', 'experiments/network_noise_support.py',
        'experiments/prepare_frozen_completion.py', 'experiments/prepare_native_joint.py',
        'experiments/audit_native_interface_support.py', 'experiments/online_stream.h',
        'experiments/audit_frozen_noise_completion.py', 'experiments/evaluate_native_joint.py',
        'experiments/evaluate_incremental_reference.py', 'scripts/dranppr_variance_components.py',
        'scripts/dranppr_linear_evidence.py', 'tests/test_network_noise_transfer.py',
        'tests/test_network_noise_support.py', 'CMakeLists.txt', 'cmake/native_local_model.cmake',
        'docs/superpowers/plans/2026-10-03-network-noise-transfer.md']
    sources = [root/n for n in names]+list((root/'experiments/native_local').glob('*'))
    for directory, suffix in [('src','.cpp'), ('include/DPGO','.h')]:
        for stem in ['Native*','Incremental*']:
            sources += list((root/directory).glob(stem+suffix))
    for source in sources:
        if source.is_file():
            target = output/'source'/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2('/tmp/dranppr_backend_build/CMakeCache.txt', output/'CMakeCache.txt')
    libraries = subprocess.check_output(['ldd', str(binaries['native-joint-audit'])], text=True)
    linked = [Path(s.split('=>',1)[1].split()[0]) for s in libraries.splitlines() if '=>' in s]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(grid=grid(), runs=[], inputs={}, all_estimators_terminal=False,
        all_predictions_terminal=False, truth_used_by_estimator=False, max_iterations=100,
        timeout_s=600, forecast_horizon_s=5, binaries={k:sha256(v) for k,v in binaries.items()},
        libraries=libraries, library_hashes={str(p):sha256(p) for p in linked if p.is_file()},
        env={k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        calibration='Same five-component fixed-linear REML; fit at first connected past; nominal training factors remain unchanged',
        scope='Centralized statistical diagnostic; no injected noise or truth variance; past-only batch checkpoints, not online per-frame benchmark')
    dump(output/'manifest.json', manifest)

    def execute(command, tag):
        start = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_600s'
        row = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(row)
        dump(output/'manifest.json', manifest)
        print(json.dumps(row), flush=True)
        if code != 0:
            raise RuntimeError('run failed; retained evidence, no parameter retry')

    solves, fits, forecasts, comparisons = [], {}, [], []
    for dataset in (1,4):
        case = output/f'd{dataset}'
        case.mkdir(exist_ok=False)
        stream = Path(f'/tmp/dranppr_causal_streams_v1/dataset{dataset}')
        metadata = json.loads((stream/'metadata.json').read_text())
        if sha256(stream/'events.jsonl') != metadata['event_sha256']:
            raise ValueError('event hash mismatch')
        shutil.copy2(stream/'events.jsonl', case/'events.jsonl')
        shutil.copy2(stream/'metadata.json', case/'metadata.json')
        warm_path = case/'warm_start.json'
        if dataset == 1:
            discovery = output/'d1_discovery'
            if not (discovery/'summary.json').exists():
                execute([str(binaries['native-joint-replay']), str(case/'events.jsonl'),
                    str(discovery), 'joint_cached', '120', *LINKS], 'd1_discovery')
            discovery_summary = json.loads((discovery/'summary.json').read_text())
            joined = discovery_summary['all_connected_elapsed_s']
            if joined is None or joined >= 30:
                raise ValueError('D1 first-connection gate not met by fixed first checkpoint')
            manifest['d1_discovery'] = dict(summary_sha256=sha256(discovery/'summary.json'),
                prefix_replayed_separately=True, cutoff_from_causal_event=True,
                note='May be a pre-run discovery; no discovery terminal state is used by any estimator or predictor')
            execute([str(binaries['native-joint-replay']), str(case/'events.jsonl'),
                str(case/'prefix'), 'joint_cached', str(joined), *LINKS], 'd1_prefix')
            execute([sys.executable,'-m','experiments.prepare_native_joint','--stream',str(stream),
                '--snapshot',str(case/'prefix/final_estimates.csv'),'--duration',str(joined),
                '--links',*LINKS,'--output',str(warm_path)], 'd1_prepare')
            prefix = read_rows(case/'prefix/causal_estimates.csv')
            original = read_rows(discovery/'causal_estimates.csv')
            if prefix != original[:len(prefix)]:
                raise ValueError('prefix causal replay mismatch')
        else:
            prior = root/'results/dranppr_full_lazy_refresh_20261003/d4/warm_start.json'
            shutil.copy2(prior, warm_path)
        warm = json.loads(warm_path.read_text())
        if warm.get('truth_used', False) or warm['provenance']['event_sha256'] != sha256(case/'events.jsonl'):
            raise ValueError('invalid common-past provenance')
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
        first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
        manifest['inputs'][str(dataset)] = dict(events_sha256=sha256(case/'events.jsonl'),
            warm_sha256=sha256(warm_path), cutoff_elapsed_s=cutoff-first)
        execute([str(binaries['native-joint-audit']),str(warm_path),str(case/'training'),
                 'central_lm','100','--linearization'], f'd{dataset}_training')
        linear = json.loads((case/'training/linearization.json').read_text())
        a, b, _ = sparse_model(linear)
        training = json.loads((case/'training/summary.json').read_text())
        errors = dict(loss=abs(.5*float(b@b)-training['final_cost']),
                      gradient=abs(float(np.linalg.norm(a.T@b))-training['posthoc_gradient_norm']))
        if max(errors.values()) > 1e-8:
            raise ValueError('native physical linearization parity failed')
        fitted = dict(nominal=dict(scales=[1.]*5, truth_used=False))
        for arm in ARMS[1:]:
            execute([sys.executable,'-m','experiments.audit_network_noise_support','fit',
                '--source',str(case/'training/linearization.json'),'--arm',arm,
                '--output',str(case/'fits'/arm)], f'd{dataset}_fit_{arm}')
            fitted[arm] = json.loads((case/'fits'/arm/'fit.json').read_text())
        fits[str(dataset)] = dict(arms=fitted, export_errors=errors)
        events = [json.loads(s) for s in (case/'events.jsonl').read_text().splitlines()]
        for elapsed in CHECKPOINTS:
            label = f'd{dataset}_t{elapsed}'
            nominal_replay = case/f'replay_{elapsed}'
            execute([str(binaries['native-joint-replay']),str(case/'events.jsonl'),str(nominal_replay),
                'joint_cached_one',str(elapsed),'--warm-start',str(warm_path),'--lazy-refresh',*LINKS], label+'_replay')
            package = prepare(events, read_rows(nominal_replay/'final_estimates.csv'), warm, elapsed, 0.)
            reference = json.loads((nominal_replay/'summary.json').read_text())
            for arm in ARMS:
                weighted = apply_scales(package, fitted[arm]['scales'], cutoff, 0.)
                package_path = case/f't{elapsed}_{arm}.json'
                dump(package_path, weighted)
                folder = case/f't{elapsed}_{arm}'
                execute([str(binaries['native-joint-audit']),str(package_path),str(folder),
                         'central_lm','100','--linearization'], label+'_'+arm)
                summary = json.loads((folder/'summary.json').read_text())
                if arm == 'nominal' and abs(summary['initial_cost']-reference['physical_cost']) > 1e-8:
                    raise ValueError('frozen replay objective mismatch')
                solves.append(dict(dataset=dataset, elapsed=elapsed, arm=arm,
                    folder=str(folder.relative_to(output)), package_sha256=sha256(package_path), summary=summary))
                if elapsed < 120:
                    start = time.perf_counter()
                    predicted = forecast(weighted, read_rows(folder/'final_estimates.csv'),
                        json.loads((folder/'linearization.json').read_text()), events, first+elapsed, 5., fitted[arm]['scales'])
                    predicted.update(dataset=dataset, elapsed=elapsed, arm=arm,
                        prediction_seconds=time.perf_counter()-start)
                    dump(folder/'forecast.json', predicted)
                    forecasts.append(dict(dataset=dataset, elapsed=elapsed, arm=arm,
                        folder=str(folder.relative_to(output)), **prediction_summary(predicted['predictions']),
                        unknown_object_observations=predicted['unknown_object_observations'],
                        covariance_solve_relative_residual=predicted['covariance_solve_relative_residual']))
                else:
                    other = case/f't{elapsed}_{arm}_gn'
                    execute([str(binaries['native-joint-audit']),str(package_path),str(other),
                             'central_gn','100'], label+'_'+arm+'_gn')
                    other_summary = json.loads((other/'summary.json').read_text())
                    comparisons.append(dict(dataset=dataset, arm=arm,
                        **state_difference(read_rows(folder/'final_estimates.csv'),read_rows(other/'final_estimates.csv')),
                        cost_abs=abs(summary['final_cost']-other_summary['final_cost'])))
        dump(output/'producer.json', dict(solves=solves, fits=fits, forecasts=forecasts, comparisons=comparisons))
    manifest['all_estimators_terminal'] = manifest['all_predictions_terminal'] = True
    dump(output/'manifest.json', manifest)


def evaluate(output, raw):
    manifest = json.loads((output/'manifest.json').read_text())
    if not manifest['all_estimators_terminal'] or not manifest['all_predictions_terminal']:
        raise ValueError('producer not completed')
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    root = Path(__file__).resolve().parents[1]
    producer = json.loads((output/'producer.json').read_text())
    results, predictions = [], []
    for dataset in (1,4):
        data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        source = audit_dataset(data, raw)
        truth = load_ground_truth(data, source.dataset_directory)
        labels = dict(data.evaluation.object_ground_truth_label_map)
        for record in producer['solves']:
            if record['dataset'] == dataset:
                quality = evaluate_joint_rows(read_rows(output/record['folder']/'final_estimates.csv'), truth, labels)
                results.append(dict(dataset=dataset, elapsed=record['elapsed'], arm=record['arm'], **quality))
        matched = None
        for arm in ARMS:
            rows = []
            by_window = []
            for elapsed in CHECKPOINTS[:-1]:
                forecast_data = json.loads((output/f'd{dataset}/t{elapsed}_{arm}/forecast.json').read_text())
                rows.extend(forecast_data['predictions'])
                by_window.append(dict(elapsed=elapsed, **prediction_summary(forecast_data['predictions'])))
            keys = [(r['robot'],r['source_row']) for r in rows]
            if len(set(keys)) != len(keys) or (matched is not None and keys != matched):
                raise ValueError('predictive support differs across arms')
            matched = keys
            by_robot = [dict(robot=r, **prediction_summary([p for p in rows if p['robot']==r]))
                        for r in sorted({p['robot'] for p in rows})]
            predictions.append(dict(dataset=dataset, arm=arm, **prediction_summary(rows),
                                    by_window=by_window, by_robot=by_robot))
    analysis = dict(results=results, predictions=predictions, comparisons=producer['comparisons'],
        truth_used_only_by_evaluation=True,
        caveats='Past-only batch checkpoint and plug-in marginal forecasts. Correlated samples, no joint likelihood or predictive parameter uncertainty. Structural identifiability and optimizer success are not calibration certificates. Split retains network-derived nonlinear chart.')
    dump(output/'analysis.json', analysis)
    for row in results:
        if row['elapsed'] == 120:
            print(json.dumps(row), flush=True)
    for row in predictions:
        print(json.dumps({k:v for k,v in row.items() if not k.startswith('by_')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run','evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.output)
    else:
        if args.raw is None:
            parser.error('--raw required for evaluation only')
        evaluate(args.output, args.raw)


if __name__ == '__main__':
    main()
