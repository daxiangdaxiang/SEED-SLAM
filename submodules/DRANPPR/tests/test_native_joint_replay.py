import csv
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture(params=['joint', 'joint_cached', 'joint_one', 'joint_cached_one'])
def mode(request):
    return request.param


def events():
    rows = []
    for k in range(8):
        for robot in [1, 2]:
            t = float(k)+.1*(robot-1)
            x, y = .2*k+robot-1, .3*(robot-1)
            rows.append(dict(robot=robot, kind='pose', pose_index=k, timestamp_s=t,
                available_time_s=t, support_max_time_s=t, from_pose_index=k-1 if k else None,
                delta=[.2, 0., 0.] if k else None, duration_s=1. if k else 0.))
            for label, point in [(6, (2., 2.)), (7, (4., -1.))]:
                if label == 7 and k < 2:
                    continue
                dx, dy = point[0]-x, point[1]-y
                rows.append(dict(robot=robot, kind='observation', pose_index=k, timestamp_s=t,
                    available_time_s=t, support_max_time_s=t, landmark_subject=label,
                    range_m=math.hypot(dx, dy), bearing_rad=math.atan2(dy, dx), source_row=2*k+label-6))
    return rows


def run(tmp_path, records, name='joint', duration=20., links=('1:2',), mode='joint'):
    binary = Path(os.environ.get('DRANPPR_NATIVE_JOINT_REPLAY_BINARY', '/tmp/dranppr_backend_build/bin/native-joint-replay'))
    assert binary.is_file(), 'build native-joint-replay first'
    stream = tmp_path/f'{name}.jsonl'
    stream.write_text(''.join(json.dumps(r)+'\n' for r in records))
    output = tmp_path/name
    process = subprocess.run([str(binary), str(stream), str(output), mode, str(duration), *links],
        capture_output=True, text=True, timeout=60)
    return process, output


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_delayed_shared_support_merges_without_freezing_other_root(tmp_path, mode):
    process, output = run(tmp_path, events(), mode=mode)
    assert process.returncode == 0, process.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert len(summary['merges']) == 1
    assert summary['merges'][0]['elapsed_s'] == 2.1
    assert summary['merges'][0]['gauge_cost_change_abs'] < 1e-8
    assert summary['all_connected_elapsed_s'] == 2.1
    assert summary['control_application_bytes'] > 0
    assert summary['optimizer_application_bytes'] > 0
    assert summary['final_components'] == 1
    assert summary['consensus_max_m'] == 0.
    rows = read(output/'final_estimates.csv')
    roots = {int(r['robot']):np.array([float(r['x']), float(r['y'])])
             for r in rows if r['kind'] == 'pose' and r['index'] == '0'}
    assert np.linalg.norm(roots[1]) == 0.
    assert np.linalg.norm(roots[2]-[1., .3]) < 1e-6
    updates = read(output/'updates.csv')
    assert all(float(r['cost_after']) <= float(r['cost_before'])+1e-8 for r in updates)
    assert all(int(r['components']) == 2 for r in updates if .1 <= float(r['elapsed_s']) < 2.1)
    if mode.endswith('_one'):
        assert max(int(r['pcg_iterations']) for r in updates) <= 1
    if 'cached' in mode:
        assert summary['response_reuses'] > 0


def test_future_frames_do_not_change_past_outputs(tmp_path, mode):
    records = events()
    p1, a = run(tmp_path, records, 'a', 3., mode=mode)
    for row in records:
        if row['timestamp_s'] > 3. and row['kind'] == 'observation':
            row['range_m'] *= 100.
    p2, b = run(tmp_path, records, 'b', 3., mode=mode)
    assert p1.returncode == p2.returncode == 0
    assert (a/'final_estimates.csv').read_bytes() == (b/'final_estimates.csv').read_bytes()
    assert (a/'causal_estimates.csv').read_bytes() == (b/'causal_estimates.csv').read_bytes()


def test_missing_support_does_not_invent_a_component_or_pose_prior(tmp_path, mode):
    records = [r for r in events() if r['kind'] == 'pose' or r['landmark_subject'] == 6]
    process, output = run(tmp_path, records, mode=mode)
    assert process.returncode == 0, process.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['final_components'] == 2
    assert summary['merges'] == []
    assert summary['all_connected_elapsed_s'] is None


def test_same_timestamp_frames_are_ordered_by_robot_not_packet_layout(tmp_path, mode):
    records = events()
    for row in records:
        row['timestamp_s'] = row['available_time_s'] = row['support_max_time_s'] = float(row['pose_index'])
    p1, a = run(tmp_path, records, 'contiguous', mode=mode)
    records.sort(key=lambda r: (r['timestamp_s'], r['kind'] != 'pose', r['robot']))
    p2, b = run(tmp_path, records, 'interleaved', mode=mode)
    assert p1.returncode == p2.returncode == 0, p2.stderr
    assert (a/'final_estimates.csv').read_bytes() == (b/'final_estimates.csv').read_bytes()


def test_equal_labels_without_observer_path_remain_separate(tmp_path, mode):
    records = events()
    for k in range(8):
        t = k+.2
        records.append(dict(robot=3, kind='pose', pose_index=k, timestamp_s=t,
            available_time_s=t, support_max_time_s=t, from_pose_index=k-1 if k else None,
            delta=[.2, 0., 0.] if k else None, duration_s=1. if k else 0.))
        for label, point in [(6,(2.,2.)),(7,(4.,-1.))]:
            dx, dy = point[0]-(.2*k+2.), point[1]-.6
            records.append(dict(robot=3,kind='observation',pose_index=k,timestamp_s=t,
                available_time_s=t,support_max_time_s=t,landmark_subject=label,
                range_m=math.hypot(dx,dy),bearing_rad=math.atan2(dy,dx),source_row=2*k+label-6))
        for robot, point in [(1,(-2.,2.)), (3,(6.,2.))]:
            t = k+.1*(robot-1)
            dx, dy = point[0]-(.2*k+robot-1),point[1]-.3*(robot-1)
            records.append(dict(robot=robot,kind='observation',pose_index=k,timestamp_s=t,
                available_time_s=t,support_max_time_s=t,landmark_subject=10,
                range_m=math.hypot(dx,dy),bearing_rad=math.atan2(dy,dx),source_row=100+k))
    records.sort(key=lambda r:(r['timestamp_s'],r['kind'] != 'pose'))
    process, output = run(tmp_path,records,links=('1:2','2:3'),mode=mode)
    assert process.returncode == 0,process.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['final_components'] == 1
    assert summary['network_identities'] == 4
    objects = [r for r in read(output/'final_estimates.csv') if r['kind']=='landmark' and r['index']=='10']
    assert abs(float(objects[0]['x'])-float(objects[1]['x'])) > 7.
    messages = [json.loads(s) for s in (output/'control_messages.jsonl').read_text().splitlines()]
    assert sum(m['application_bytes'] for m in messages) == summary['control_application_bytes']
    assert all(abs(m['from']-m['to']) == 1 for m in messages)


def test_inexact_noisy_updates_preserve_descent_without_claiming_linear_convergence(tmp_path, mode):
    records = events()
    for row in records:
        if row['kind'] == 'observation':
            row['range_m'] += .03*math.sin(row['source_row']+row['robot'])
            row['bearing_rad'] += .01*math.cos(row['source_row'])
    process, output = run(tmp_path,records,mode=mode)
    assert process.returncode == 0, process.stderr
    rows = read(output/'updates.csv')
    assert all(float(r['cost_after']) <= float(r['cost_before'])+1e-10 for r in rows)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['consensus_max_m'] == 0.
    for field in ['snapshot', 'linearization', 'contribution']:
        measured = sum(float(r[f'update_{field}_ms']) for r in rows)/1000.
        assert measured > 0.
        assert summary[f'update_{field}_seconds'] == pytest.approx(measured)
    nested = sum(summary[f'update_{field}_seconds'] for field in ['snapshot', 'linearization', 'contribution'])
    assert nested <= summary['model_seconds']+summary['publication_seconds']+1e-8
    if mode.endswith('_one'):
        assert summary['pcg_max_relative_residual'] > 1e-5
        assert all(int(r['pcg_iterations']) <= 1 for r in rows)
    else:
        assert summary['pcg_max_relative_residual'] <= 1e-8
