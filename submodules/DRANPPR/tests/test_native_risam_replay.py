"""Native-factor bridge tests; set DRANPPR_NATIVE_RISAM_BINARY to opt in."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_incremental import IncrementalPoint2Robot


@pytest.fixture
def binary():
    path = os.environ.get('DRANPPR_NATIVE_RISAM_BINARY')
    if not path:
        pytest.skip('explicit upstream build required')
    assert Path(path).is_file(), 'native replay executable has not been built'
    return path


def pose(t, index, delta=None):
    return dict(robot=1, kind='pose', timestamp_s=t, available_time_s=t,
                support_max_time_s=t, pose_index=index,
                from_pose_index=index-1 if index else None,
                delta=delta, duration_s=1. if index else 0.)


def observation(t, index, row, distance, bearing=0., subject=6):
    return dict(robot=1, kind='observation', timestamp_s=t, available_time_s=t,
                support_max_time_s=t, pose_index=index, source_row=row,
                landmark_subject=subject, range_m=distance, bearing_rad=bearing)


def frames():
    return [[pose(0., 0), observation(0., 0, 0, 2.),
             observation(0., 0, 1, 2., np.pi/2, 7)],
            [pose(1., 1, [1., 0., 0.]), observation(1., 1, 2, 1.),
             observation(1., 1, 3, np.sqrt(5), np.arctan2(2., -1.), 7)],
            [pose(2., 2, [0., 1., .1]), observation(2., 2, 4, np.sqrt(2), -.1-np.pi/4)]]


def run(binary, root, data, mode='isam2', duration=10., success=True):
    root.mkdir()
    events = root/'events.jsonl'
    events.write_text(''.join(json.dumps(p)+'\n' for frame in data for p in frame))
    out = root/'result'
    result = subprocess.run([binary, str(events), str(out), mode, str(duration)],
                            capture_output=True, text=True, timeout=30)
    if success:
        assert result.returncode == 0, result.stderr
        return {name: list(csv.DictReader((out/f'{name}.csv').open()))
                for name in ['causal_estimates', 'final_estimates', 'updates']}
    assert result.returncode != 0
    return result.stderr


@pytest.mark.parametrize('mode', ['isam2', 'risam', 'risam_gn'])
def test_exact_motion_and_atomic_insertion(binary, tmp_path, mode):
    result = run(binary, tmp_path/mode, frames(), mode)
    assert len(result['updates']) == 3
    assert [int(r['new_factors']) for r in result['updates']] == [3, 3, 2]
    last = result['causal_estimates'][-1]
    np.testing.assert_allclose([float(last[k]) for k in ['x','y','yaw']], [1.,1.,.1], atol=1e-8)
    assert len(result['final_estimates']) == 5


def test_cpp_gaussian_matches_python_same_physical_factors(binary, tmp_path):
    data = frames()
    data[1][1]['range_m'] = 1.13
    data[2][1]['bearing_rad'] += .04
    result = run(binary, tmp_path/'parity', data)
    robot = IncrementalPoint2Robot(1)
    for frame, row in zip(data, result['causal_estimates']):
        robot.push_batch(frame)
        p = robot.graph.estimate().atPose2(gtsam.symbol('x', frame[0]['pose_index']))
        np.testing.assert_allclose([float(row[k]) for k in ['x','y','yaw']],
                                   [p.x(),p.y(),p.theta()], atol=1e-9)


def test_simultaneous_robots_do_not_split_a_sensor_frame(binary, tmp_path):
    second = dict(pose(0., 0), robot=2)
    data = [[pose(0., 0)], [second], [observation(0., 0, 0, 2.)]]
    result = run(binary, tmp_path/'simultaneous', data)
    assert len(result['updates']) == 2
    assert [int(r['new_factors']) for r in result['updates']] == [2, 1]


@pytest.mark.parametrize('mode', ['isam2', 'risam'])
def test_future_cannot_change_causal_prefix(binary, tmp_path, mode):
    data = frames()
    a = run(binary, tmp_path/'a', data, mode, duration=1.)
    data[-1][1]['range_m'] = 20.
    b = run(binary, tmp_path/'b', data, mode, duration=1.)
    assert a['causal_estimates'] == b['causal_estimates']
    assert len(a['causal_estimates']) == 2


@pytest.mark.parametrize('corruption,expected', [
    ('duplicate','duplicate observation'), ('future','noncausal'),
    ('parent','parent'), ('range','range'), ('order','time decreased')])
def test_rejects_invalid_stream(binary, tmp_path, corruption, expected):
    data = frames()
    if corruption == 'duplicate': data[1][1]['source_row'] = 0
    if corruption == 'future': data[1][1]['support_max_time_s'] = 2.
    if corruption == 'parent': data[1][0]['from_pose_index'] = 5
    if corruption == 'range': data[1][1]['range_m'] = -1.
    if corruption == 'order': data = [data[0], data[1], data[0]]
    error = run(binary, tmp_path/corruption, data, success=False)
    assert expected in error
