"""Paired same-chart nonlinear replay; no truth or noise fitting enters a solve."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def paired_rows(left, right, filename, identity):
    a, b = read_rows(left/filename), read_rows(right/filename)
    if len(a) != len(b):
        raise ValueError(f'{filename}: different rows')
    for x, y in zip(a, b):
        if any(x[k] != y[k] for k in identity):
            raise ValueError(f'{filename}: different row identity')
    return list(zip(a,b))


def compare_runs(left, right):
    a, b = [json.loads((p/'summary.json').read_text()) for p in [left,right]]
    result = dict(graph_equal=all(a[k] == b[k] for k in
        ['frames','physical_factors','prefix_physical_factors']),
        physical_cost_abs=abs(a['physical_cost']-b['physical_cost']),
        application_bytes_difference=b['application_bytes']-a['application_bytes'],
        service_speedup=a['service_seconds']/b['service_seconds'],
        setup_inclusive_speedup=(a['service_seconds']+a['setup_seconds'])/
            (b['service_seconds']+b['setup_seconds']))
    for name in ['final','causal']:
        pairs = paired_rows(left,right,f'{name}_estimates.csv',['robot','kind','index'])
        errors = [abs(float(x[k])-float(y[k])) for x,y in pairs for k in ['x','y','yaw'] if x[k] and y[k]]
        result[f'{name}_state_max_abs'] = max(errors, default=0.)
    pairs = paired_rows(left,right,'updates.csv',['robot','timestamp_s'])
    result['frame_mismatches'] = {k:sum(x[k] != y[k] for x,y in pairs) for k in
        ['accepted','pcg_iterations','optimizer_bytes','control_bytes','refreshed_keys','factors_linearized']}
    for k in ['cost_before','cost_after']:
        result[f'{k}_max_abs'] = max(abs(float(x[k])-float(y[k])) for x,y in pairs)
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True, help='Absolute elapsed stream cutoff, not suffix length')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', choices=['tight','one'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    binary = args.output/'native-joint-replay'
    shutil.copy2(args.binary,binary)
    shutil.copy2(args.warm_start,args.output/'warm_start.json')
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root/'CMakeLists.txt', source_root/'cmake/native_local_model.cmake',
               source_root/'experiments/online_stream.h', Path(__file__)]
    sources += list((source_root/'experiments/native_local').glob('*'))
    for name in ['IncrementalLocalModel','IncrementalConditionalModel','NativeSchurModel',
                 'NativeLocalResponse','NativeLocalOptimizer','NativeInterface','NativeBetweenPose2Factor']:
        sources += [source_root/f'include/DPGO/{name}.h',source_root/f'src/{name}.cpp']
    for source in sources:
        if not source.is_file():
            continue
        target = args.output/'source'/source.relative_to(source_root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest = dict(scope='Short common-past nonlinear suffix, reference/candidate/candidate/reference; not independent seeds',
        binary_sha256=sha256(binary), events=str(args.events), events_sha256=sha256(args.events),
        warm_start_sha256=sha256(args.warm_start), duration=args.duration,budget=args.budget,
        env={k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        libraries=subprocess.check_output(['ldd',str(binary)],text=True),runs=[])
    for arm in ['reference1','candidate1','candidate2','reference2']:
        mode = 'joint_cached' if arm.startswith('reference') else 'joint_conditional_cached'
        if args.budget == 'one':
            mode += '_one'
        command = [str(binary.resolve()),str(args.events.resolve()),str((args.output/arm).resolve()),mode,
            str(args.duration),'--warm-start',str((args.output/'warm_start.json').resolve()),'1:2','2:3','3:4','4:5']
        start = time.perf_counter()
        with (args.output/f'{arm}.log').open('w') as log:
            try:
                code = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_180s'
        record = dict(arm=arm,command=command,exit_code=code,wall_seconds=time.perf_counter()-start)
        manifest['runs'].append(record)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if code != 0:
            raise RuntimeError(f'{arm} failed; see retained log')
    comparisons = [compare_runs(args.output/f'reference{i}',args.output/f'candidate{i}') for i in [1,2]]
    (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False)+'\n')
    print(json.dumps(comparisons),flush=True)


if __name__ == '__main__':
    main()
