import json

import pytest

from test_native_conditional_replay import rows, run
from test_native_joint_audit import package
from test_native_matched_online import stream


@pytest.mark.parametrize('suffix', ['', '_one'])
def test_lazy_refresh_preserves_physical_descent_and_conditional_parity(tmp_path, suffix):
    outputs = []
    for mode in ['joint_cached', 'joint_conditional_cached']:
        process, out = run(tmp_path/mode, mode+suffix, lazy=True)
        assert process.returncode == 0, process.stderr
        outputs.append(out)
    summaries = [json.loads((p/'summary.json').read_text()) for p in outputs]
    for summary in summaries:
        assert summary['lazy_refresh'] is True
        assert summary['chart_refresh_tolerance'] is None
        assert summary['root_displacement'] == 0
        assert summary['consensus_max_m'] < 1e-12
        assert summary['physical_factors'] == 30
        assert summary['truth_used'] is False
    a, b = summaries
    assert b['physical_cost'] == pytest.approx(a['physical_cost'], abs=1e-8)
    assert a['refreshed_keys'] == b['refreshed_keys']
    for out, summary in zip(outputs, summaries):
        updates = rows(out/'updates.csv')
        assert int(updates[0]['refresh_rounds']) == 0
        for previous, current in zip(updates, updates[1:]):
            assert int(current['refresh_rounds']) == int(previous['refresh_requests'])
        assert sum(int(r['refresh_requests']) for r in updates) == summary['refresh_requests']
        assert sum(int(r['refresh_rounds']) for r in updates) == summary['refresh_rounds']
        assert summary['refresh_pending'] == bool(int(updates[-1]['refresh_requests']))
        for row in updates:
            assert float(row['cost_after']) <= float(row['cost_before'])+1e-10
            assert int(row['pcg_iterations']) <= (1 if suffix else 500)
            if row['full_step_decrease'] != 'null':
                assert bool(int(row['refresh_requests'])) == (float(row['full_step_decrease']) <= 0.)
            else:
                assert int(row['refresh_requests']) == 1
    for filename in ['final_estimates.csv', 'causal_estimates.csv']:
        left, right = [rows(p/filename) for p in outputs]
        for x, y in zip(left, right):
            for field in ['x', 'y', 'yaw']:
                if x[field]:
                    assert float(x[field]) == pytest.approx(float(y[field]), abs=1e-8)


def test_lazy_refresh_is_causal_and_not_unconditionally_refreshing(tmp_path):
    data = package()
    events = stream(data)
    p, a = run(tmp_path/'a', 'joint_conditional_cached_one', events=events, duration=3., lazy=True)
    assert p.returncode == 0, p.stderr
    for event in events:
        if event['timestamp_s'] > 3. and event['kind'] == 'observation':
            event['range_m'] += 10.
    p, b = run(tmp_path/'b', 'joint_conditional_cached_one', events=events, lazy=True)
    assert p.returncode == 0, p.stderr
    left = (a/'causal_estimates.csv').read_text().splitlines()
    assert left == (b/'causal_estimates.csv').read_text().splitlines()[:len(left)]
    updates = rows(b/'updates.csv')
    assert any(int(r['refresh_rounds']) == 0 for r in updates)


@pytest.mark.parametrize('mode,warm', [('joint', True), ('joint_cached', False)])
def test_lazy_refresh_rejects_unsupported_execution_paths(tmp_path, mode, warm):
    process, _ = run(tmp_path/'run', mode, warm=warm, lazy=True)
    assert process.returncode != 0
    assert 'lazy refresh requires cached common-past' in process.stderr
