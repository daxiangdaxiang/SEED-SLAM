import copy
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_native_joint_audit import package


def run(tmp_path, data):
    binary = Path(os.environ.get('DRANPPR_CONDITIONAL_BINARY',
                                '/tmp/dranppr_backend_build/bin/native-conditional-audit'))
    assert binary.is_file(), 'build native-conditional-audit first'
    source, output = tmp_path/'input.json', tmp_path/'audit.json'
    source.write_text(json.dumps(data))
    process = subprocess.run([str(binary), str(source), str(output)],
                             text=True, capture_output=True, timeout=90)
    return process, output


def test_native_prefixes_and_factor_edits_match_fresh_response(tmp_path):
    process, output = run(tmp_path, package())
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result['scope'] == 'frozen_chart_incremental_response_not_online_estimator'
    assert result['truth_used'] is False
    assert len(result['robots']) == 2
    for robot in result['robots']:
        assert robot['valid_prefixes'] > 0
        assert robot['max_product_relative_error'] < 1e-9
        assert robot['max_gradient_relative_error'] < 1e-9
        assert robot['max_recovery_relative_error'] < 1e-8
        assert robot['max_private_relative_residual'] < 1e-8
        assert robot['factors_final'] == 11
        assert len(robot['edits']) == 2
        assert {row['name'] for row in robot['edits']} == {'late_factor', 'early_factor'}
        assert robot['conditional_update_seconds'] > 0
        assert robot['reference_update_seconds'] > 0
    assert result['routed']['relative_residual'] < 1e-8
    assert result['routed']['step_relative_error'] < 1e-8


def test_unobservable_prefixes_are_reported_not_regularized(tmp_path):
    data = package()
    second = data['robots'][1]
    second['observations'] = [o for o in second['observations'] if o['pose'] > 0]
    process, output = run(tmp_path, data)
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    robot = next(r for r in result['robots'] if r['robot'] == 2)
    assert robot['singular_prefixes'] >= 1
    assert any(row['status'] == 'singular_private_system' for row in robot['prefixes'])


@pytest.mark.parametrize('mutation', ['chart', 'source', 'topology'])
def test_invalid_packages_are_rejected(tmp_path, mutation):
    data = copy.deepcopy(package())
    if mutation == 'chart':
        data['robots'][1]['objects'][0]['point'][0] += 1
    elif mutation == 'source':
        data['robots'][0]['observations'][1]['source'] = 0
    else:
        data['links'] = []
    process, _ = run(tmp_path, data)
    assert process.returncode != 0
