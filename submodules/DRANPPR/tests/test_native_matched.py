import copy
import json
import math
import os
from pathlib import Path
import subprocess

import pytest
from test_native_joint_audit import package, run as run_dran, rows


def run(tmp_path, data, method, rounds=10, budget=0):
    binary = Path(os.environ.get('DRANPPR_MATCHED_BINARY',
                                '/tmp/dranppr_matched_build/native-matched-baseline'))
    assert binary.is_file(), 'build native-matched-baseline first'
    source = tmp_path/f'{method}.json'
    output = tmp_path/method
    source.write_text(json.dumps(data))
    process = subprocess.run([str(binary), str(source), str(output), method,
                              str(rounds), str(budget)], capture_output=True,
                             text=True, timeout=90)
    return process, output


@pytest.mark.parametrize('method', ['imesa', 'imesa_dogleg', 'imesa_paper', 'rimesa', 'rimesa_gaussian'])
def test_same_physical_graph_gauge_and_ledger(tmp_path, method):
    data = package()
    process, output = run(tmp_path, data, method)
    assert process.returncode == 0, process.stderr
    summary = json.loads((output/'summary.json').read_text())
    reference, refdir = run_dran(tmp_path, data, 'routed', 1)
    assert reference.returncode == 0, reference.stderr
    ref = json.loads((refdir/'summary.json').read_text())
    assert summary['initial_cost'] == pytest.approx(ref['initial_cost'], abs=1e-10)
    assert summary['physical_factors'] == 22
    assert summary['gauge_priors'] == 1
    assert summary['root_displacement'] < 1e-9
    assert summary['nonroot_root_displacement_max'] > .01
    assert summary['rounds'] == 10
    assert math.isfinite(summary['final_cost'])
    assert summary['service_seconds'] >= summary['startup_seconds'] > 0
    messages = [json.loads(s) for s in (output/'messages.jsonl').read_text().splitlines()]
    assert len(messages) == summary['messages'] == 40
    assert {m['kind'] for m in messages} == {'declaration', 'solution'}
    assert all({m['from'], m['to']} == {1, 2} for m in messages)
    assert all(m['application_bytes'] == m['payload_bytes']+40 for m in messages)
    assert sum(m['application_bytes'] for m in messages) == summary['application_bytes']
    assert sum(m['payload_bytes'] for m in messages) == summary['payload_bytes']
    assert len(rows(output)) == 12


def test_budget_stops_at_complete_handshake_round(tmp_path):
    probe_dir = tmp_path/'probe'
    probe_dir.mkdir()
    process, folder = run(probe_dir, package(), 'imesa', rounds=1)
    assert process.returncode == 0, process.stderr
    size = json.loads((folder/'summary.json').read_text())['application_bytes']
    process, folder = run(tmp_path, package(), 'imesa', rounds=100, budget=2*size)
    assert process.returncode == 0, process.stderr
    summary = json.loads((folder/'summary.json').read_text())
    assert summary['rounds'] == 2
    assert summary['application_bytes'] <= 2*size
    assert summary['stop_reason'] == 'byte_budget'


@pytest.mark.parametrize('mutation', ['chart', 'source', 'gauge', 'timestamp', 'roster', 'link'])
def test_invalid_package_is_rejected(tmp_path, mutation):
    data = copy.deepcopy(package())
    if mutation == 'chart':
        data['robots'][1]['objects'][0]['point'][0] += .1
    elif mutation == 'source':
        data['robots'][0]['observations'][1]['source'] = 0
    elif mutation == 'gauge':
        data['root_robot'] = 9
    elif mutation == 'timestamp':
        data['robots'][1]['poses'][1]['timestamp_s'] = 99.
    elif mutation == 'roster':
        data['clusters'][0]['observers'] = [1]
    else:
        data['links'] = []
    process, _ = run(tmp_path, data, 'imesa')
    assert process.returncode != 0


def test_ordinary_consensus_recovers_small_noiseless_fixture(tmp_path):
    process, output = run(tmp_path, package(), 'imesa', rounds=2000)
    assert process.returncode == 0, process.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['consensus_max_m'] < 1e-6
    # Upstream default relinearization is approximate, not an exact batch solve.
    assert summary['final_cost'] < 1e-3


@pytest.mark.parametrize('method', ['imesa', 'imesa_paper', 'rimesa'])
def test_common_rigid_gauge_does_not_change_result(tmp_path, method):
    data = package()
    process, refdir = run(tmp_path, data, method)
    assert process.returncode == 0, process.stderr
    for robot in data['robots']:
        for p in robot['poses']:
            x, y, yaw = p['value']
            p['value'] = [-y+3., x-7., yaw+math.pi/2]
        for p in robot['objects']:
            x, y = p['point']
            p['point'] = [-y+3., x-7.]
    moved = tmp_path/'moved'
    moved.mkdir()
    process, outdir = run(moved, data, method)
    assert process.returncode == 0, process.stderr
    for a, b in zip(rows(refdir), rows(outdir)):
        assert float(b['x']) == pytest.approx(-float(a['y'])+3., abs=1e-8)
        assert float(b['y']) == pytest.approx(float(a['x'])-7., abs=1e-8)
