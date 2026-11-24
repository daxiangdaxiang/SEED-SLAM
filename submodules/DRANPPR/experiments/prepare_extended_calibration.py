"""Prepare additional known-identity sequences under the unchanged prefix fit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.audit_mean_evidence import sha256, write_json

CASES = [(2,120.,'dranppr_lazy_transfer_stress_20261003/d2_clean'),
         (3,120.,'dranppr_lazy_transfer_stress_20261003/d3_clean'),
         (9,300.,'dranppr_full_lazy_refresh_20261003/d9')]


def run(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = root/'build/native/bin/native-joint-audit'
    shutil.copy2(binary, output/'native.binary')
    manifest = dict(benchmark_cases=[dict(dataset=d,duration=t) for d,t,_ in CASES],
        runs=[], inputs={}, all_training_terminal=False, truth_used=False,
        binary=dict(path=str(binary),sha256=sha256(binary)),
        scope='Additional previously inspected sequences, same first-prefix joint_qr fit; not untouched data. No future measurements or GT in fitting.')
    for name in ('experiments/prepare_extended_calibration.py', 'experiments/audit_network_noise_support.py',
                 'experiments/network_noise_support.py','scripts/dranppr_variance_components.py',
                 'scripts/dranppr_linear_evidence.py'):
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)

    def execute(command, tag):
        begun = time.perf_counter()
        with (output/(tag+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout_600s'
        record = dict(tag=tag, command=command, exit_code=code, seconds=time.perf_counter()-begun)
        manifest['runs'].append(record)
        write_json(output/'manifest.json', manifest)
        print(json.dumps(record),flush=True)
        return code == 0

    success = True
    for dataset, duration, name in CASES:
        original = root/'results'/name
        # Archived per-case ledgers bind both copied files; never trust path alone.
        ledger = next(p/'SHA256SUMS' for p in (original,original.parent) if (p/'SHA256SUMS').exists())
        hashes = dict(line.split('  ',1)[::-1] for line in ledger.read_text().splitlines())
        case = output/f'd{dataset}'
        case.mkdir()
        for name in ('events.jsonl','warm_start.json'):
            path = original/name
            key = str(path.relative_to(ledger.parent))
            if sha256(path) != hashes.get(key):
                raise ValueError('archived input checksum mismatch: '+key)
            shutil.copy2(path, case/name)
        warm = json.loads((case/'warm_start.json').read_text())
        if (warm.get('truth_used',False) or warm['provenance']['dataset'] != dataset or
                warm['provenance']['event_sha256'] != sha256(case/'events.jsonl')):
            raise ValueError('invalid warm input provenance')
        first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
        if cutoff-first >= duration:
            raise ValueError('duration does not extend prefix')
        manifest['inputs'][str(dataset)] = dict(events_sha256=sha256(case/'events.jsonl'),
            warm_sha256=sha256(case/'warm_start.json'), source=str(original),
            cutoff_elapsed_s=cutoff-first, original_ledger_sha256=sha256(ledger))
        trained = execute([str(binary),str(case/'warm_start.json'),str(case/'training'),
            'central_lm','100','--linearization'],f'd{dataset}_training')
        if trained:
            trained = execute([sys.executable,'-m','experiments.audit_network_noise_support','fit',
                '--source',str(case/'training/linearization.json'),'--arm','joint_qr',
                '--output',str(case/'fits/joint_qr')],f'd{dataset}_fit')
        success &= trained
    manifest['all_training_terminal'] = True
    manifest['all_training_successful'] = success
    write_json(output/'manifest.json', manifest)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))
    if not success:
        raise RuntimeError('calibration failed; retained all attempted cases without parameter substitution')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
