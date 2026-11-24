"""Same-prefix native backend controls with matched application-byte ceilings."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_baseline(folder, package, budget, reference_cost):
    summary = json.loads((folder/'summary.json').read_text())
    if not math.isclose(summary['initial_cost'], reference_cost, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('initial physical likelihood mismatch')
    expected = sum(len(r['odometry'])+len(r['observations']) for r in package['robots'])
    if summary['physical_factors'] != expected or summary['gauge_priors'] != 1:
        raise ValueError('physical factor/gauge count mismatch')
    if summary['root_displacement'] > 1e-8:
        raise ValueError('root gauge changed')
    messages = [json.loads(s) for s in (folder/'messages.jsonl').read_text().splitlines()]
    edges = {frozenset(e) for e in package['links']}
    for row in messages:
        if frozenset((row['from'], row['to'])) not in edges:
            raise ValueError('out-of-topology message')
        if row['application_bytes'] != row['payload_bytes']+40:
            raise ValueError('header accounting mismatch')
    if len(messages) != summary['messages'] or len(messages) != 4*len(edges)*summary['rounds']:
        raise ValueError('incomplete round/handshake')
    for column in ('application_bytes', 'payload_bytes'):
        if sum(r[column] for r in messages) != summary[column]:
            raise ValueError('byte sum mismatch')
    if summary['application_bytes'] > budget:
        raise ValueError('budget exceeded')
    summary['ledger_verified'] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--baseline', type=Path, default=Path('/tmp/dranppr_matched_build/native-matched-baseline'))
    parser.add_argument('--dran', type=Path, default=Path('/tmp/dranppr_backend_build/bin/native-joint-audit'))
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--method', action='append', choices=['imesa', 'imesa_dogleg',
        'imesa_paper', 'rimesa', 'rimesa_gaussian'])
    args = parser.parse_args()
    if any(k < 1 or k > 100 for k in args.steps) or len(set(args.steps)) != len(args.steps):
        raise ValueError('unique routed step caps in 1..100 required')
    package = json.loads(args.package.read_text())
    if package.get('truth_used', False):
        raise ValueError('truth-assisted package is not a causal control')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(args.package, args.output/'input.json')
    manifest = dict(package_sha256=sha(args.package), truth_used_by_estimator=False,
        scope='Common primal start, cold baseline internal states; frozen prefix, not online',
        binaries={}, sources={}, runs=[], all_estimators_terminal=False)
    for tag, binary in [('baseline', args.baseline), ('dran', args.dran)]:
        libraries = subprocess.check_output(['ldd', str(binary)], text=True)
        linked = [Path(s.split('=>', 1)[1].split()[0]) for s in libraries.splitlines() if '=>' in s]
        manifest['binaries'][tag] = dict(path=str(binary), sha256=sha(binary), ldd=libraries,
            library_sha256={str(p):sha(p) for p in linked if p.is_file()})
        shutil.copy2(binary, args.output/f'{tag}.binary')
    cache = args.baseline.parent/'CMakeCache.txt'
    if cache.is_file():
        shutil.copy2(cache, args.output/'baseline_CMakeCache.txt')
        config = {}
        for line in cache.read_text().splitlines():
            if '=' in line and not line.startswith(('#', '//')):
                key, value = line.split('=', 1)
                config[key.split(':', 1)[0]] = value
        manifest['upstream'] = {}
        for key in ['IMESA_SOURCE_DIR', 'RIMESA_SOURCE_DIR', 'RISAM_SOURCE_DIR']:
            source = Path(config[key])
            rev = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            status = subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True)
            tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files'], text=True).splitlines()
            hashes = {}
            for name in tracked:
                path = source/name
                if path.suffix in ('.cpp', '.h', '.hpp') and path.is_file():
                    hashes[name] = sha(path)
                    dest = args.output/'source_snapshot'/key/name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
            manifest['upstream'][key] = dict(path=str(source), commit=rev, status=status, source_sha256=hashes)
        manifest['static_libraries'] = {key: dict(path=config[key], sha256=sha(config[key]))
            for key in ['RIMESA_LIBRARY', 'RISAM_LIBRARY']}
    for relative in ['experiments/audit_native_matched.py',
                     'experiments/native_matched/baseline.cpp',
                     'experiments/native_matched/CMakeLists.txt',
                     'src/NativeBetweenPose2Factor.cpp', 'include/DPGO/NativeBetweenPose2Factor.h',
                     'tests/test_native_matched.py', 'tests/test_audit_native_matched.py']:
        path = root/relative
        destination = args.output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest['sources'][relative] = sha(path)
    def execute(tag, command):
        start = time.perf_counter()
        with (args.output/f'{tag}.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_120s'
        row = dict(tag=tag, command=command, exit_code=code, wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(row)
        dump(args.output/'manifest.json', manifest)
        print(json.dumps(row), flush=True)
        return code == 0
    results = []
    for steps in args.steps:
        tag = f'routed_{steps}'
        folder = args.output/tag
        if not execute(tag, [str(args.dran), str(args.output/'input.json'), str(folder), 'routed', str(steps)]):
            continue
        summary = json.loads((folder/'summary.json').read_text())
        budget = summary['application_bytes']
        count = sum(len(r['odometry'])+len(r['observations']) for r in package['robots'])
        summary.update(tag=tag, matched_budget=budget, physical_factors=count)
        results.append(summary)
        for method in (args.method or ['imesa_paper', 'rimesa', 'rimesa_gaussian']):
            tag = f'{method}_b{steps}'
            folder = args.output/tag
            if not execute(tag, [str(args.baseline), str(args.output/'input.json'), str(folder),
                                 method, '10000', str(budget)]):
                continue
            stats = verify_baseline(folder, package, budget, summary['initial_cost'])
            stats.update(tag=tag, matched_budget=budget)
            results.append(stats)
    manifest['all_estimators_terminal'] = True
    dump(args.output/'manifest.json', manifest)
    if args.raw:
        # Ground truth is loaded only after every estimator process has exited.
        sys.path.insert(0, str(root))
        from benchmarks.mrclam.evaluate import load_ground_truth
        from benchmarks.mrclam.source import audit_dataset, load_manifest
        from experiments.evaluate_incremental_reference import read_rows
        from experiments.evaluate_native_joint import evaluate_joint_rows, package_rows
        dataset = package['provenance']['dataset']
        data_manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        source = audit_dataset(data_manifest, args.raw)
        truth = load_ground_truth(data_manifest, source.dataset_directory)
        labels = dict(data_manifest.evaluation.object_ground_truth_label_map)
        initial_quality = evaluate_joint_rows(package_rows(package), truth, labels)
        for row in results:
            row['quality'] = evaluate_joint_rows(read_rows(args.output/row['tag']/'final_estimates.csv'), truth, labels)
    else:
        initial_quality = None
    dump(args.output/'comparison.json', dict(results=results, initial_quality=initial_quality,
        warning='Costs of disagreeing local copies are not feasible joint costs; robust targets also differ. Read cost, consensus and state quality together.'))
    with (args.output/'summary.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=['tag','budget','bytes','physical_cost','consensus_m',
            'setup_s','service_s','trajectory_rmse_m','object_rmse_m'])
        writer.writeheader()
        for row in results:
            q = row.get('quality', {})
            writer.writerow(dict(tag=row['tag'],budget=row['matched_budget'],bytes=row['application_bytes'],
                physical_cost=row['final_cost'],consensus_m=row['consensus_max_m'],setup_s=row['setup_seconds'],
                service_s=row.get('service_seconds', row.get('solve_seconds')),
                trajectory_rmse_m=q.get('translation_rmse_m'),object_rmse_m=q.get('object_copy_rmse_m')))
    if any(r['exit_code'] != 0 for r in manifest['runs']):
        raise SystemExit('Some runs failed; see retained manifest and logs')


if __name__ == '__main__':
    main()
