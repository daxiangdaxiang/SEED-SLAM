import json
from pathlib import Path
import subprocess
import sys

from test_native_joint_audit import package


def test_matched_runner_retains_complete_manifest_and_caps(tmp_path):
    script = Path(__file__).resolve().parents[1]/'experiments/audit_native_matched.py'
    assert script.is_file(), 'implement matched runner'
    source = tmp_path/'fixture.json'
    source.write_text(json.dumps(package()))
    output = tmp_path/'audit'
    result = subprocess.run([sys.executable, str(script), '--package', str(source),
                             '--output', str(output), '--steps', '1'],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(manifest['runs']) == 4
    assert all(row['exit_code'] == 0 for row in manifest['runs'])
    assert manifest['truth_used_by_estimator'] is False
    results = json.loads((output/'comparison.json').read_text())['results']
    assert len(results) == 4
    assert {r['mode'] for r in results} == {'routed', 'imesa_paper', 'rimesa', 'rimesa_gaussian'}
    baseline = [r for r in results if r['mode'] != 'routed']
    assert all(r['application_bytes'] <= r['matched_budget'] for r in baseline)
    assert all(r['ledger_verified'] for r in baseline)
    assert len({r['physical_factors'] for r in results}) == 1
