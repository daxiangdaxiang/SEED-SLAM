"""Run matched frozen joint controls sequentially; preserve every exit/log."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import time

from experiments.audit_native_checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True, help='NAME=package.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for item in args.case:
        name, source = item.split('=', 1)
        if not name.replace('_', '').isalnum() or name in cases:
            raise ValueError('invalid or duplicate case name')
        cases[name] = Path(source)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [root/'experiments/native_local/joint.cpp', root/'experiments/native_local/joint_step.h', root/'src/NativeInterface.cpp',
        root/'include/DPGO/NativeInterface.h', root/'src/NativeSchurModel.cpp',
        root/'src/IncrementalLocalModel.cpp', root/'src/NativeLocalOptimizer.cpp',
        root/'src/NativeBetweenPose2Factor.cpp', root/'experiments/prepare_native_joint.py', Path(__file__)]
    sources += [root/'include/DPGO'/name for name in ['IncrementalLocalModel.h', 'NativeSchurModel.h',
        'NativeBetweenPose2Factor.h', 'NativeLocalOptimizer.h']]
    sources += [root/'tests'/name for name in ['test_native_joint_prepare.py', 'test_native_joint_audit.py',
        'test_evaluate_native_joint.py', 'testNativeInterface.cpp']]
    sources += [root/'experiments/evaluate_native_joint.py', root/'cmake/native_local_model.cmake']
    for source in sources:
        target = args.output/'source_snapshot'/source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    libraries = subprocess.check_output(['ldd', str(args.binary)], text=True)
    linked = [Path(line.split('=>', 1)[1].split()[0]) for line in libraries.splitlines() if '=>' in line]
    manifest = dict(scope='Frozen synchronous joint controls; not online or robust adaptation',
        binary_sha256=sha256(args.binary), source_hashes={str(p.relative_to(root)):sha256(p) for p in sources},
        dynamic_libraries=libraries, dynamic_library_hashes={str(p):sha256(p) for p in linked if p.is_file()},
        input_hashes={name:sha256(p) for name, p in cases.items()}, runs=[])
    for name, source in cases.items():
        for mode in ['routed', 'central_gn', 'central_lm']:
            output = args.output/f'{name}_{mode}'
            command = [str(args.binary), str(source), str(output), mode, '100']
            start = time.perf_counter()
            with (args.output/f'{name}_{mode}.log').open('x') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout_600s'
            manifest['runs'].append(dict(case=name, mode=mode, command=command, exit_code=code,
                wall_seconds=time.perf_counter()-start))
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps(manifest['runs'][-1]), flush=True)


if __name__ == '__main__':
    main()
