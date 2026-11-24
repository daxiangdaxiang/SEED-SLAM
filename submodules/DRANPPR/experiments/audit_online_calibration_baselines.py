"""Common-past causal native noise-transfer comparison with pinned riMESA."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_native_online import verify
from experiments.evaluate_incremental_reference import read_rows


def benchmark_cases(manifest, datasets=None):
    declared = manifest.get('benchmark_cases', [{'dataset':1,'duration':120.},
                                                {'dataset':4,'duration':120.}])
    result = []
    try:
        for case in declared:
            dataset, duration = case['dataset'], float(case['duration'])
            if (type(dataset) is not int or dataset not in (1,2,3,4,9) or
                    not math.isfinite(duration) or duration <= 0 or
                    dataset in [d for d,_ in result]):
                raise ValueError('invalid benchmark cases')
            result.append((dataset,duration))
    except (KeyError, TypeError) as error:
        raise ValueError('invalid benchmark cases') from error
    if not result:
        raise ValueError('empty benchmark cases')
    if datasets is not None:
        if (not datasets or len(set(datasets)) != len(datasets) or
                not set(datasets).issubset(d for d,_ in result)):
            raise ValueError('invalid requested dataset subset')
        result = [(d,t) for d,t in result if d in datasets]
    return result


def calibration_for_case(source, dataset, manifest):
    path = source/f'd{dataset}/fits/joint_qr/fit.json'
    failed = [r for r in manifest.get('runs',[]) if r.get('tag') in
              (f'd{dataset}_fit',f'd{dataset}_training') and r.get('exit_code') != 0]
    if failed:
        return None, str(failed[-1]['exit_code'])
    if not path.is_file():
        raise ValueError('missing calibration without recorded failure')
    fit = json.loads(path.read_text())
    if (not fit.get('success',False) or fit.get('truth_used',False) or
            len(fit['scales']) != 5 or any(not math.isfinite(v) or v <= 0 for v in fit['scales'])):
        raise ValueError('invalid calibration')
    return fit, None


def common_credits(left, right):
    if not left or len(left) != len(right):
        raise ValueError('different/empty frame support')
    total_a = total_b = 0
    result = []
    for a, b in zip(left, right):
        key = int(a['robot']), float(a['timestamp_s'])
        if key != (int(b['robot']), float(b['timestamp_s'])):
            raise ValueError('different frame support')
        increments = [int(r[k]) for r in (a,b) for k in ('optimizer_bytes','control_bytes')]
        if min(increments) < 0:
            raise ValueError('negative bytes')
        total_a += sum(increments[:2])
        total_b += sum(increments[2:])
        result.append(dict(robot=key[0], timestamp_s=key[1], application_bytes=min(total_a,total_b)))
    return result


def run(source, output, datasets=None):
    source, output = source.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[1]
    source_manifest = json.loads((source/'manifest.json').read_text())
    cases = benchmark_cases(source_manifest,datasets)
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(source/name) != digest:
            raise ValueError('source archive changed: '+name)
        hashes[name] = digest
    output.mkdir(parents=True, exist_ok=False)
    native = root/'build/native/bin/native-joint-replay'
    baseline = root/'build/matched/native-matched-baseline'
    binaries = {}
    for name, path in [('native',native), ('baseline',baseline)]:
        libraries = subprocess.check_output(['ldd',str(path)],text=True)
        if str(root/'build/research-deps/gtsam-imesa-build/gtsam/libgtsam') not in libraries or 'libtbb' in libraries:
            raise ValueError('unexpected baseline/runtime dependency')
        linked = [Path(s.split('=>',1)[1].split()[0]) for s in libraries.splitlines() if '=>' in s]
        binaries[name] = dict(path=str(path), sha256=sha256(path), ldd=libraries,
                             library_hashes={str(p):sha256(p) for p in linked if p.is_file()})
        shutil.copy2(path,output/(name+'.binary'))
    names = ['experiments/audit_online_calibration_baselines.py','experiments/audit_native_online.py',
             'experiments/native_local/joint_replay.cpp','experiments/native_matched/CMakeLists.txt',
             'experiments/native_matched/baseline.cpp','experiments/native_matched/online.h',
             'experiments/online_stream.h','tests/test_native_suffix_scales.py',
             'tests/test_online_calibration_baselines.py','docs/superpowers/plans/2026-10-29-online-baseline-validation.md']
    for name in names:
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    dependencies = {}
    for name, path in [('rimesa',root/'build/research-deps/rimesa-src'),
                       ('risam',root/'build/research-deps/risam-repaired'),
                       ('imesa',Path('/home/jackal/zyx/dpgo_ICRA/baselines/iMESA_deps/imesa_clean'))]:
        dependencies[name] = dict(commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','-C',str(path),'status','--short'],text=True), files={})
        for file in path.rglob('*'):
            if not file.is_file() or file.suffix not in ('.h','.cpp') or '.git' in file.parts:
                continue
            relative = file.relative_to(path)
            target = output/'dependencies'/name/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,target)
            dependencies[name]['files'][str(relative)] = sha256(file)
    for name, path in [('baseline',root/'build/matched/CMakeCache.txt'),
                       ('native',root/'build/native/CMakeCache.txt')]:
        shutil.copy2(path,output/(name+'_CMakeCache.txt'))
    manifest = dict(source=str(source), source_hashes=hashes, binaries=binaries, runs=[],
        benchmark_cases=[dict(dataset=d,duration=t) for d,t in cases],
        dependencies=dependencies,
        all_estimators_terminal=False, truth_used_by_estimator=False, timeout_s=600,
        scope='Native Pose2/Point2 persistent suffix, common primal warm start and cold internal solvers; externally fitted frozen scales, not continuous distributed learning',
        comparison='One sweep/frame and per-frame minimum native-arm byte ceiling. No equal-time claim; calibration cost/communication excluded from suffix ledger.',
        inputs={}, unavailable=[])
    write_json(output/'manifest.json',manifest)

    def execute(command,tag):
        started = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_600s'
        row = dict(tag=tag,command=command,exit_code=code,seconds=time.perf_counter()-started)
        manifest['runs'].append(row)
        write_json(output/'manifest.json',manifest)
        print(json.dumps(row),flush=True)
        return code == 0

    results = []
    for dataset, duration in cases:
        case = output/f'd{dataset}'
        case.mkdir()
        original = source/f'd{dataset}'
        for name in ('events.jsonl','warm_start.json'):
            shutil.copy2(original/name,case/name)
        package = json.loads((case/'warm_start.json').read_text())
        fit, failure = calibration_for_case(source,dataset,source_manifest)
        if (package.get('truth_used',False) or
                package['provenance']['event_sha256'] != sha256(case/'events.jsonl')):
            raise ValueError('invalid input provenance')
        first = min(r['poses'][0]['timestamp_s'] for r in package['robots'])
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in package['robots'])
        if package['provenance']['dataset'] != dataset or duration <= cutoff-first:
            raise ValueError('dataset/duration does not extend declared prefix')
        if failure:
            manifest['unavailable'].append(dict(dataset=dataset,arm='calibrated',reason=failure,
                consequence='No candidate substituted. Baseline byte ceiling uses nominal only.'))
        manifest['inputs'][str(dataset)] = dict(scales=fit['scales'] if fit else None,
            fit_sha256=sha256(original/'fits/joint_qr/fit.json') if fit else None,
            package_sha256=sha256(case/'warm_start.json'),events_sha256=sha256(case/'events.jsonl'))
        arms = ('nominal','calibrated') if fit else ('nominal',)
        for arm in arms:
            command = [str(native),str(case/'events.jsonl'),str(case/arm),'joint_cached_one',str(duration),
                       '--warm-start',str(case/'warm_start.json'),'--lazy-refresh']
            if arm == 'calibrated':
                command += ['--suffix-variance-multipliers',*[str(v) for v in fit['scales']]]
            command += [f'{a}:{b}' for a,b in package['links']]
            if not execute(command,f'd{dataset}_{arm}'):
                raise RuntimeError('native arm failed; cannot manufacture common budget')
        updates = {a:read_rows(case/a/'updates.csv') for a in arms}
        credits = common_credits(updates['nominal'],updates.get('calibrated',updates['nominal']))
        (case/'credits.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in credits))
        reference = json.loads((case/'nominal/summary.json').read_text())
        for arm in arms:
            results.append(dict(dataset=dataset,arm=arm,folder=str((case/arm).relative_to(output)),
                                summary=verify(case/arm,package,reference,credits)))
        for mode in ('imesa_paper','rimesa'):
            for schedule in ('one','bytes'):
                arm = mode+'_'+schedule
                command = [str(baseline),str(case/'warm_start.json'),str(case/arm),mode,'--online',
                           str(case/'events.jsonl'),str(duration),'1' if schedule=='one' else str(case/'credits.jsonl')]
                if execute(command,f'd{dataset}_{arm}'):
                    results.append(dict(dataset=dataset,arm=arm,folder=str((case/arm).relative_to(output)),
                                        summary=verify(case/arm,package,reference,credits)))
        write_json(output/'producer.json',results)
    manifest['all_estimators_terminal'] = True
    write_json(output/'manifest.json',manifest)


def evaluate(output,raw):
    manifest = json.loads((output/'manifest.json').read_text())
    if not manifest['all_estimators_terminal']:
        raise ValueError('estimation incomplete')
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset,load_manifest
    from experiments.evaluate_native_joint import evaluate_joint_rows
    from experiments.analyze_joint_delivery import delivery_metrics
    root = Path(__file__).resolve().parents[1]
    records = json.loads((output/'producer.json').read_text())
    for dataset, _ in benchmark_cases(manifest):
        data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        source = audit_dataset(data,raw)
        truth = load_ground_truth(data,source.dataset_directory)
        labels = dict(data.evaluation.object_ground_truth_label_map)
        support = None
        for row in records:
            if row['dataset'] != dataset:
                continue
            folder = output/row['folder']
            final = read_rows(folder/'final_estimates.csv')
            causal = read_rows(folder/'causal_estimates.csv')
            keys = [(r['robot'],r['kind'],r['index']) for r in final]
            if support is not None and keys != support:
                raise ValueError('different final state support')
            support = keys
            row['quality'] = evaluate_joint_rows(final,truth,labels)
            row['causal_quality'] = evaluate_joint_rows(causal,truth,labels)
            row['delivery'] = delivery_metrics(causal,read_rows(folder/'updates.csv'),truth,row['causal_quality'])
            print(json.dumps(dict(dataset=dataset,arm=row['arm'],quality=row['quality'],
                causal_rmse=row['causal_quality']['translation_rmse_m'],
                MB=row['summary']['application_bytes']/1e6,service_s=row['summary']['service_seconds'])),flush=True)
    write_json(output/'comparison.json',dict(records=records,truth_used_only_for_evaluation=True,
        failures=[r for r in manifest['runs'] if r['exit_code']!=0],
        unavailable=manifest.get('unavailable',[]),
        warning='Common primal start, cold internal states; fixed known identities. No riMESA predictive covariance calibration claim. Suffix-only accounting, not full system runtime or total deployment communication.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','evaluate'])
    parser.add_argument('--source',type=Path,default=Path('results/dranppr_network_noise_transfer_qr_20261003'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--datasets',type=int,nargs='+',help='Run only these declared source datasets')
    args=parser.parse_args()
    if args.mode=='run':
        run(args.source,args.output,args.datasets)
    else:
        if args.raw is None:
            parser.error('--raw required for evaluation')
        evaluate(args.output,args.raw)
