import json
from pathlib import Path
import subprocess
import sys
import pytest

from test_native_matched_online import package, stream


@pytest.mark.parametrize('mode', ['joint_cached', 'joint_one'])
def test_online_audit_records_equal_information_and_credit_caps(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    script = root/'experiments/audit_native_online.py'
    assert script.is_file(), 'implement online audit runner'
    source, events, output = tmp_path/'input.json', tmp_path/'events.jsonl', tmp_path/'out'
    source.write_text(json.dumps(package()))
    events.write_text(''.join(json.dumps(e)+'\n' for e in stream(package())))
    result = subprocess.run([sys.executable, '-m', 'experiments.audit_native_online',
        '--package', str(source), '--events', str(events), '--output', str(output), '--duration', '4',
        '--dran-mode', mode],
        cwd=root, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(manifest['runs']) == 7
    assert manifest['all_estimators_terminal']
    assert all(r['exit_code'] == 0 for r in manifest['runs'])
    results = json.loads((output/'comparison.json').read_text())['results']
    assert len({r['physical_factors'] for r in results}) == 1
    assert len({r['frames'] for r in results}) == 1
    assert all(r['ledger_verified'] for r in results)
    assert all(r['information_verified'] for r in results)
