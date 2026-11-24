import csv
import json
import os
import subprocess

import pytest

from test_native_joint_audit import package
from test_native_matched_online import stream


def run(folder, mode, data=None, events=None, duration=4., warm=True, audit=False, lazy=False, extra=()):
    folder.mkdir()
    data = data or package()
    (folder/'input.json').write_text(json.dumps(data))
    (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in (events or stream(data))))
    binary = os.environ.get('DRANPPR_NATIVE_JOINT_REPLAY_BINARY',
                            '/tmp/dranppr_backend_build/bin/native-joint-replay')
    command = [binary, str(folder/'events.jsonl'), str(folder/'result'), mode, str(duration)]
    if warm:
        command += ['--warm-start', str(folder/'input.json')]
    if audit:
        command += ['--chart-audit']
    if lazy:
        command += ['--lazy-refresh']
    command += list(extra)
    command += [f'{a}:{b}' for a, b in data['links']]
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return process, folder/'result'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize('suffix', ['', '_one'])
def test_conditional_nonlinear_replay_matches_same_chart_normal_model(tmp_path, suffix):
    outputs = []
    for name in ['joint_cached', 'joint_conditional_cached']:
        process, out = run(tmp_path/name, name+suffix)
        assert process.returncode == 0, process.stderr
        outputs.append(out)
    a, b = [json.loads((p/'summary.json').read_text()) for p in outputs]
    for key in ['frames', 'physical_factors', 'prefix_physical_factors', 'factors_linearized',
                'refreshed_keys', 'application_bytes', 'pcg_iterations', 'control_application_bytes']:
        assert a[key] == b[key], key
    assert b['physical_factors'] == 30
    assert b['conditional_response'] is True
    assert b['conditionals_rebuilt'] > 0
    assert b['root_displacement'] == 0
    assert b['consensus_max_m'] < 1e-12
    assert b['physical_cost'] == pytest.approx(a['physical_cost'], rel=1e-10, abs=1e-10)
    for filename in ['final_estimates.csv', 'causal_estimates.csv']:
        left, right = [rows(p/filename) for p in outputs]
        assert len(left) == len(right)
        for x, y in zip(left, right):
            for field in x:
                if field in ['x', 'y', 'yaw'] and x[field]:
                    assert float(y[field]) == pytest.approx(float(x[field]), abs=1e-8)
                else:
                    assert x[field] == y[field]
    left, right = [rows(p/'updates.csv') for p in outputs]
    for x, y in zip(left, right):
        assert x['accepted'] == y['accepted']
        assert float(y['cost_after']) <= float(y['cost_before'])+1e-10
        assert float(y['cost_after']) == pytest.approx(float(x['cost_after']), abs=1e-9)
    assert sum(int(r['conditionals_rebuilt']) for r in right) == b['conditionals_rebuilt']


def test_conditional_cache_uses_no_future_measurements(tmp_path):
    data = package()
    events = stream(data)
    p, a = run(tmp_path/'a', 'joint_conditional_cached_one', events=events, duration=3.)
    assert p.returncode == 0, p.stderr
    for event in events:
        if event['timestamp_s'] > 3. and event['kind'] == 'observation':
            event['range_m'] += 10.
    p, b = run(tmp_path/'b', 'joint_conditional_cached_one', events=events)
    assert p.returncode == 0, p.stderr
    left = (a/'causal_estimates.csv').read_text().splitlines()
    assert left == (b/'causal_estimates.csv').read_text().splitlines()[:len(left)]


def test_conditional_prototype_does_not_claim_cold_start_support(tmp_path):
    p, _ = run(tmp_path/'cold', 'joint_conditional_cached', warm=False)
    assert p.returncode != 0
    assert 'warm-start' in p.stderr
