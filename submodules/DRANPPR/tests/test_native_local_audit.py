"""Executable contracts for the native nonlinear diagnostic; no Python GTSAM."""
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from test_native_risam_checkpoint import inputs


def execute(tmp_path, change=None):
    binary = os.environ.get('DRANPPR_NATIVE_LOCAL_BINARY')
    if not binary:
        pytest.skip('explicit optional native audit binary required')
    assert Path(binary).is_file(), 'native local audit executable has not been built'
    events, state = inputs()
    if change:
        change(events, state)
    stream, snapshot, output = tmp_path/'events.jsonl', tmp_path/'state.json', tmp_path/'out'
    stream.write_text(''.join(json.dumps(row)+'\n' for row in events))
    snapshot.write_text(json.dumps(state))
    result = subprocess.run([binary, str(stream), str(snapshot), str(output), '1', '1'],
                            capture_output=True, text=True, timeout=30)
    return result, output


def test_same_likelihood_small_noisy_prefix(tmp_path):
    result, output = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['truth_used'] is False
    assert summary['poses'] == 2
    assert summary['observations'] == 2
    assert summary['initial_cost'] == pytest.approx(6.12, abs=1e-10)
    assert summary['initial_directional_fd'] == pytest.approx(summary['initial_exact_gradient_norm'], abs=1e-5)
    assert len(summary['runs']) == 5
    costs = []
    for run in summary['runs']:
        assert run['cost'] <= summary['initial_cost']
        assert run['exact_gradient_norm'] < 1e-4
        assert abs(run['physical_directional_fd']) < 1e-4
        costs.append(run['cost'])
        rows = list(csv.DictReader((output/(run['method']+'_estimates.csv')).open()))
        root = next(r for r in rows if r['kind'] == 'pose' and int(r['index']) == 0)
        assert max(abs(float(root[k])) for k in ['x', 'y', 'yaw']) < 1e-10
    assert max(costs)-min(costs) < 1e-8


def test_object_covariance_matches_analytic_range_information(tmp_path):
    result, output = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    covariances = []
    for method in ['exact_schur', 'exact_full', 'exact_lm']:
        path = output/(method+'_object_covariance.json')
        assert path.is_file(), 'export conditional native object uncertainty'
        record = json.loads(path.read_text())
        assert record['object_ids'] == [6]
        assert record['fixed_root_index'] == 0
        assert record['truth_used'] is False
        covariance = np.asarray(record['covariance'])
        assert covariance.shape == (2, 2)
        assert np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
        assert np.linalg.eigvalsh(covariance).min() > 0
        # x-information: two range factors (100 each), eliminate pose x (200).
        assert covariance[0, 0] == pytest.approx(1./150., abs=1e-10)
        assert record['information_inverse_relative_residual'] < 1e-10
        covariances.append(covariance)
    assert max(np.linalg.norm(c-covariances[0]) for c in covariances) < 1e-8


@pytest.mark.parametrize('corruption,expected', [
    ('root', 'root gauge'), ('missing', 'keys'), ('extra', 'keys'),
    ('timestamp', 'timestamp'), ('duplicate', 'duplicate observation'),
    ('future', 'noncausal')])
def test_rejects_invalid_prefix(tmp_path, corruption, expected):
    def change(events, state):
        if corruption == 'root': state[0]['x'] = 1.
        if corruption == 'missing': state.pop()
        if corruption == 'extra': state.append(dict(state[-1], index=7))
        if corruption == 'timestamp': state[1]['timestamp_s'] = .5
        if corruption == 'duplicate': events[3]['source_row'] = 0
        if corruption == 'future': events[3]['support_max_time_s'] = 2.
    result, _ = execute(tmp_path, change)
    assert result.returncode != 0
    assert expected in result.stderr
