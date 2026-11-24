import copy
import csv
import json
import math
from pathlib import Path
import subprocess

import pytest
from test_native_joint_audit import package


def stream(data):
    events = []
    for r in data['robots']:
        labels = {o['cluster']: o['label'] for o in r['objects']}
        for p in r['poses']:
            k, t = p['index'], p['timestamp_s']
            base = dict(robot=r['id'], pose_index=k, timestamp_s=t,
                        available_time_s=t, support_max_time_s=t)
            odom = next((o for o in r['odometry'] if o['target'] == k), None)
            events.append(dict(base, kind='pose', from_pose_index=k-1 if k else None,
                delta=odom['delta'] if k else None, duration_s=odom['dt'] if k else 0.))
            for o in r['observations']:
                if o['pose'] == k:
                    events.append(dict(base, kind='observation', source_row=o['source'],
                        landmark_subject=labels[o['cluster']], range_m=o['range'], bearing_rad=o['bearing']))
        for k in [3, 4]:
            t = float(k)
            base = dict(robot=r['id'], pose_index=k, timestamp_s=t,
                        available_time_s=t, support_max_time_s=t)
            events.append(dict(base, kind='pose', from_pose_index=k-1,
                delta=[.4*math.cos(.2*(r['id']-1)), -.4*math.sin(.2*(r['id']-1)), 0.], duration_s=1.))
            # A new shared label must not collide with frozen cluster numbering.
            x, y, yaw = .4*k+r['id']-1, r['id']-1, .2*(r['id']-1)
            events.append(dict(base, kind='observation', source_row=100+k,
                landmark_subject=20, range_m=math.hypot(3-x, 4-y), bearing_rad=math.atan2(4-y, 3-x)-yaw))
    return sorted(events, key=lambda e: (e['timestamp_s'], e['robot'], e['kind'] != 'pose'))


def run(folder, method, data=None, events=None, duration=4., rounds='1'):
    folder.mkdir()
    data = data or package()
    (folder/'input.json').write_text(json.dumps(data))
    (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in (events or stream(data))))
    out = folder/'result'
    if method in ['dran', 'dran_conditional']:
        mode = 'joint_conditional_cached' if method == 'dran_conditional' else 'joint_cached'
        command = ['/tmp/dranppr_backend_build/bin/native-joint-replay', str(folder/'events.jsonl'),
                   str(out), mode, str(duration), '--warm-start', str(folder/'input.json')]
        command += [f'{a}:{b}' for a, b in data['links']]
    else:
        command = ['/tmp/dranppr_matched_build/native-matched-baseline', str(folder/'input.json'),
                   str(out), method, '--online', str(folder/'events.jsonl'), str(duration), rounds]
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return process, out


@pytest.mark.parametrize('method', ['dran', 'dran_conditional', 'imesa_paper', 'rimesa'])
def test_persistent_suffix_graph_and_new_label(tmp_path, method):
    process, out = run(tmp_path/method, method)
    assert process.returncode == 0, process.stderr
    s = json.loads((out/'summary.json').read_text())
    assert s['frames'] == 4
    assert s['physical_factors'] == 30
    assert s['prefix_physical_factors'] == 22
    assert s['root_displacement'] < 1e-8
    assert s['warm_start_cutoff_s'] == 2.
    assert s['truth_used'] is False
    assert s['setup_seconds'] > 0
    assert s['service_seconds'] >= s['startup_seconds']
    with (out/'final_estimates.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    assert sum(r['kind'] == 'landmark' and int(r['index']) == 20 for r in rows) == 2
    with (out/'causal_estimates.csv').open() as f:
        causal = list(csv.DictReader(f))
    assert [(int(r['robot']), int(r['index'])) for r in causal] == [(1, 3), (2, 3), (1, 4), (2, 4)]
    with (out/'updates.csv').open() as f:
        updates = list(csv.DictReader(f))
    assert len(updates) == 4
    assert math.isfinite(s['physical_cost'])


@pytest.mark.parametrize('method', ['dran', 'dran_conditional', 'imesa_paper', 'rimesa'])
def test_future_measurements_do_not_change_past(tmp_path, method):
    data = package()
    process, first = run(tmp_path/'short', method, data, duration=3.)
    assert process.returncode == 0, process.stderr
    events = stream(data)
    for e in events:
        if e['kind'] == 'observation' and e['timestamp_s'] > 3.:
            e['range_m'] += 2.
    process, longer = run(tmp_path/'long', method, data, events=events)
    assert process.returncode == 0, process.stderr
    a = (first/'causal_estimates.csv').read_text().splitlines()
    b = (longer/'causal_estimates.csv').read_text().splitlines()
    assert a == b[:len(a)]


@pytest.mark.parametrize('method', ['dran', 'dran_conditional', 'imesa_paper', 'rimesa'])
def test_mismatched_past_is_not_silently_skipped(tmp_path, method):
    data = package()
    events = stream(data)
    next(e for e in events if e['kind'] == 'observation')['range_m'] += .01
    process, _ = run(tmp_path/method, method, data, events)
    assert process.returncode != 0
    assert 'prefix' in process.stderr.lower()


@pytest.mark.parametrize('method', ['imesa_paper', 'rimesa'])
def test_cumulative_byte_ceiling_includes_startup(tmp_path, method):
    process, dran = run(tmp_path/'dran', 'dran')
    assert process.returncode == 0, process.stderr
    with (dran/'updates.csv').open() as f:
        updates = list(csv.DictReader(f))
    credits, total = [], 0
    for row in updates:
        total += int(row['optimizer_bytes'])+int(row['control_bytes'])
        credits.append(dict(robot=int(row['robot']), timestamp_s=float(row['timestamp_s']), application_bytes=total))
    schedule = tmp_path/'credits.jsonl'
    schedule.write_text(''.join(json.dumps(r)+'\n' for r in credits))
    process, out = run(tmp_path/method, method, rounds=str(schedule))
    assert process.returncode == 0, process.stderr
    trace = [json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
    assert all(r['application_bytes'] <= c['application_bytes'] for r, c in zip(trace, credits))
    messages = [json.loads(s) for s in (out/'messages.jsonl').read_text().splitlines()]
    summary = json.loads((out/'summary.json').read_text())
    assert sum(r['application_bytes'] for r in messages) == summary['application_bytes']
    assert len(messages) == 4*summary['rounds']
    assert summary['application_bytes'] == trace[-1]['application_bytes']


@pytest.mark.parametrize('method', ['dran', 'dran_conditional', 'imesa_paper', 'rimesa'])
def test_new_observer_joins_previously_split_label(tmp_path, method):
    data = package()
    third = copy.deepcopy(data['robots'][1])
    third['id'] = 3
    data['robots'].append(third)
    data['links'] = [[1, 2], [2, 3]]
    for c in data['clusters']:
        c['observers'] = [1, 2, 3]
    for r, cluster in [(data['robots'][0], 10), (third, 11)]:
        r['objects'].append(dict(cluster=cluster, label=20, point=[3., 4.]))
        p = r['poses'][-1]['value']
        r['observations'].append(dict(pose=2, cluster=cluster, source=50,
            range=math.hypot(3-p[0], 4-p[1]), bearing=math.atan2(4-p[1], 3-p[0])-p[2]))
        data['clusters'].append(dict(cluster=cluster, label=20, owner=r['id'], observers=[r['id']]))
    process, out = run(tmp_path/method, method, data)
    assert process.returncode == 0, process.stderr
    with (out/'final_estimates.csv').open() as f:
        labels = [r for r in csv.DictReader(f) if r['kind'] == 'landmark' and int(r['index']) == 20]
    assert len(labels) == 3
    summary = json.loads((out/'summary.json').read_text())
    assert summary['physical_factors'] == 47


@pytest.mark.parametrize('method', ['dran', 'dran_conditional', 'imesa_paper', 'rimesa'])
def test_neighbor_label_cannot_be_silently_reinterpreted_from_split_prefix(tmp_path, method):
    data = package()
    data['robots'][1]['objects'][0]['cluster'] = 3
    for o in data['robots'][1]['observations']:
        if o['cluster'] == 0:
            o['cluster'] = 3
    data['clusters'][0]['observers'] = [1]
    data['clusters'].append(dict(cluster=3, label=6, owner=2, observers=[2]))
    process, _ = run(tmp_path/method, method, data)
    assert process.returncode != 0
    assert 'identity' in process.stderr.lower()
