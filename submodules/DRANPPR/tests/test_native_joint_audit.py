import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def package():
    robots = []
    objects = [[2., 3.], [4., -1.], [-1., 2.]]
    for r in [1, 2]:
        poses, observations = [], []
        for k in [0, 1, 2]:
            x, y, yaw = .4*k+(r-1), float(r-1), .2*(r-1)
            poses.append(dict(index=k, timestamp_s=float(k),
                value=[x+(.12 if r == 2 else .02*k), y-.03*k, yaw+.04*k]))
            for c, (ox, oy) in enumerate(objects):
                observations.append(dict(pose=k, cluster=c, range=math.hypot(ox-x, oy-y),
                    bearing=math.atan2(oy-y, ox-x)-yaw, source=3*k+c))
        robots.append(dict(id=r, poses=poses,
            objects=[dict(cluster=c, label=6+c, point=[p[0]+.1, p[1]-.05]) for c, p in enumerate(objects)],
            odometry=[dict(source=k-1, target=k,
                delta=[.4*math.cos(.2*(r-1)), -.4*math.sin(.2*(r-1)), 0.], dt=1.) for k in [1, 2]],
            observations=observations))
    return dict(schema='dranppr_frozen_joint_v1', root_robot=1, links=[[1, 2]], robots=robots,
        clusters=[dict(cluster=c, label=6+c, owner=1, observers=[1, 2]) for c in range(3)])


def run(tmp_path, data, mode, iterations=100):
    binary = Path(os.environ.get('DRANPPR_NATIVE_JOINT_BINARY', '/tmp/dranppr_backend_build/bin/native-joint-audit'))
    assert binary.is_file(), 'build native-joint-audit first'
    target = tmp_path/f'{mode}_{iterations}'
    source = tmp_path/f'{mode}_{iterations}.json'
    source.write_text(json.dumps(data))
    process = subprocess.run([str(binary), str(source), str(target), mode, str(iterations)],
                             capture_output=True, text=True, timeout=90)
    return process, target


def rows(target):
    with (target/'final_estimates.csv').open() as stream:
        return list(csv.DictReader(stream))


def positions(target):
    return np.array([[float(r['x']), float(r['y'])] for r in rows(target)])


def test_routed_full_first_step_and_joint_optimum(tmp_path):
    outputs = {}
    for method, iterations in [('routed', 1), ('central_gn', 1),
                               ('routed', 100), ('central_gn', 100), ('central_lm', 100)]:
        process, output = run(tmp_path, package(), method, iterations)
        assert process.returncode == 0, process.stderr
        summary = json.loads((output/'summary.json').read_text())
        assert summary['final_cost'] < summary['initial_cost']
        assert summary['consensus_max_m'] == 0.
        assert summary['root_displacement'] < 1e-10
        assert summary['nonroot_root_displacement_max'] > .05
        outputs[method, iterations] = output
    assert np.max(np.abs(positions(outputs['routed', 1])-positions(outputs['central_gn', 1]))) < 1e-8
    for method in ['central_gn', 'central_lm']:
        assert np.max(np.abs(positions(outputs['routed', 100])-positions(outputs[method, 100]))) < 1e-6
    stats = json.loads((outputs['routed', 100]/'summary.json').read_text())
    assert stats['application_bytes'] > stats['payload_bytes'] > 0
    assert stats['scalar_collective_bytes'] > 0
    assert stats['pcg_max_relative_residual'] < 1e-8
    assert stats['final_cost'] < 1e-12


@pytest.mark.parametrize('mutation', ['chart', 'timestamp', 'source', 'topology', 'identity'])
def test_malformed_joint_inputs_are_rejected(tmp_path, mutation):
    data = package()
    if mutation == 'chart':
        data['robots'][1]['objects'][0]['point'][0] += .1
    elif mutation == 'timestamp':
        data['robots'][1]['poses'][1]['timestamp_s'] = 9.
    elif mutation == 'source':
        data['robots'][1]['observations'][1]['source'] = 0
    elif mutation == 'topology':
        data['links'] = []
    else:
        data['clusters'][0]['observers'] = [1]
    process, _ = run(tmp_path, data, 'routed')
    assert process.returncode != 0


def test_common_global_gauge_does_not_change_physical_solution(tmp_path):
    data = package()
    process, reference = run(tmp_path, data, 'routed')
    assert process.returncode == 0, process.stderr
    rotation = np.array([[0., -1.], [1., 0.]])
    shift = np.array([3., -7.])
    for robot in data['robots']:
        for pose in robot['poses']:
            pose['value'][:2] = (rotation @ pose['value'][:2]+shift).tolist()
            pose['value'][2] += math.pi/2
        for point in robot['objects']:
            point['point'] = (rotation @ point['point']+shift).tolist()
    moved = tmp_path/'moved'
    moved.mkdir()
    process, transformed = run(moved, data, 'routed')
    assert process.returncode == 0, process.stderr
    assert np.max(np.abs(positions(transformed)-(positions(reference) @ rotation.T+shift))) < 1e-7
