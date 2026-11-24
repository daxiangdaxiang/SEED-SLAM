"""Pinned C++ completion contracts, without loading the system Python GTSAM."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def binary():
    path = os.environ.get('DRANPPR_NATIVE_CHECKPOINT_BINARY')
    if not path:
        pytest.skip('explicit pinned checkpoint build required')
    return path


def inputs():
    events = []
    for index in range(3):
        events.append(dict(robot=1, kind='pose', timestamp_s=float(index),
                           available_time_s=float(index), support_max_time_s=float(index),
                           pose_index=index, from_pose_index=index-1 if index else None,
                           delta=[1., 0., 0.] if index else None,
                           duration_s=1. if index else 0.))
        events.append(dict(robot=1, kind='observation', timestamp_s=float(index),
                           available_time_s=float(index), support_max_time_s=float(index),
                           pose_index=index, source_row=index, landmark_subject=6,
                           range_m=2. if index == 0 else 1.12, bearing_rad=0.))
    state = [dict(robot=1, kind='pose', index=0, timestamp_s=0., x=0., y=0., yaw=0.),
             dict(robot=1, kind='pose', index=1, timestamp_s=1., x=.7, y=0., yaw=0.),
             dict(robot=1, kind='landmark', index=6, x=2., y=0.)]
    return events, state


def execute(binary, tmp_path, mode='sig_final', change=None):
    assert Path(binary).is_file(), 'native checkpoint executable has not been built'
    events, state = inputs()
    if change:
        change(events, state)
    stream, snapshot, output = tmp_path/'events.jsonl', tmp_path/'state.json', tmp_path/'out'
    stream.write_text(''.join(json.dumps(row)+'\n' for row in events))
    snapshot.write_text(json.dumps(state))
    result = subprocess.run([binary, str(stream), str(snapshot), str(output), '1', '1', mode],
                            capture_output=True, text=True, timeout=30)
    return result, output


@pytest.mark.parametrize('mode', ['gaussian', 'sig_final'])
def test_completes_identical_prefix_and_preserves_gauge(binary, tmp_path, mode):
    result, output = execute(binary, tmp_path, mode)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['poses'] == 2
    assert summary['observations'] == 2
    assert summary['truth_used'] is False
    assert summary['after']['cost'] <= summary['before']['cost'] + 1e-10
    assert summary['after']['gradient_norm'] < 1e-4
    # Ordinary odometry stays quadratic; only the two polar observations are robust.
    r2, c2 = 1.8**2, summary['shape']**2
    expected_obs = .5*r2 if mode == 'gaussian' else .5*c2*r2/(c2+r2)
    assert summary['before']['odometry_cost'] == pytest.approx(4.5, abs=1e-11)
    assert summary['before']['observation_cost'] == pytest.approx(expected_obs, abs=1e-11)
    assert summary['before']['cost'] == pytest.approx(4.5+expected_obs, abs=1e-11)
    if mode == 'sig_final':
        assert summary['official_sig_cost_max_difference'] < 1e-12
        assert summary['official_sig_weight_max_difference'] < 1e-12
    rows = list(csv.DictReader((output/'after_estimates.csv').open()))
    root = next(r for r in rows if r['kind'] == 'pose' and int(r['index']) == 0)
    assert max(abs(float(root[k])) for k in ['x', 'y', 'yaw']) < 1e-10
    assert all(math.isfinite(float(r['x'])) for r in rows)


@pytest.mark.parametrize('corruption,expected', [
    ('root', 'root gauge'), ('missing', 'keys'), ('extra', 'keys'),
    ('timestamp', 'timestamp'), ('duplicate', 'duplicate observation'),
    ('future', 'noncausal')])
def test_rejects_incompatible_checkpoint(binary, tmp_path, corruption, expected):
    def change(events, state):
        if corruption == 'root': state[0]['x'] = 1.
        if corruption == 'missing': state.pop()
        if corruption == 'extra': state.append(dict(state[-1], index=7))
        if corruption == 'timestamp': state[1]['timestamp_s'] = .5
        if corruption == 'duplicate': events[3]['source_row'] = 0
        if corruption == 'future': events[3]['support_max_time_s'] = 2.
    result, _ = execute(binary, tmp_path, change=change)
    assert result.returncode != 0
    assert expected in result.stderr
