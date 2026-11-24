import copy
import math

import numpy as np
import pytest

from experiments.prepare_native_joint import prepare


def fixture():
    events, states = [], []
    for robot in [1, 2, 3]:
        events.append(dict(kind='pose', robot=robot, pose_index=0, timestamp_s=0.,
                           support_max_time_s=0., delta=None, duration_s=0., from_pose_index=None))
        states.append(dict(kind='pose', robot=robot, index=0, timestamp_s=0.,
                           x=0., y=0., yaw=0.))
        labels = [6, 7, 8] if robot != 2 else [6, 7]
        for index, label in enumerate(labels):
            events.append(dict(kind='observation', robot=robot, pose_index=0,
                timestamp_s=0., support_max_time_s=0., landmark_subject=label,
                source_row=index, range_m=1., bearing_rad=0.))
            states.append(dict(kind='landmark', robot=robot, index=label,
                               x=float(label-6)+10*robot, y=2.*robot))
    return events, states


def test_common_charts_and_topology_restricted_identity():
    events, states = fixture()
    result = prepare(events, states, [(1, 2), (2, 3)], 1.)
    assert result['root_robot'] == 1
    assert result['truth_used'] is False
    assert len(result['clusters']) == 4
    assert [c['observers'] for c in result['clusters'] if c['label'] == 8] == [[1], [3]]
    robots = result['robots']
    assert robots[1]['poses'][0]['value'] == pytest.approx([-10., -2., 0.])
    shared = [[o['point'] for o in r['objects'] if o['label'] == 6][0] for r in robots]
    assert np.max(np.abs(np.array(shared)-shared[0])) == 0
    assert not any('fixed' in r for r in robots)
    assert states[0]['x'] == 0.


def test_unsupported_bootstrap_and_future_state_rejected():
    events, states = fixture()
    for state in states:
        if state['kind'] == 'landmark' and state['robot'] == 2:
            state['x'] = state['y'] = 0.
    with pytest.raises(ValueError, match='bootstrap support'):
        prepare(events, states, [(1, 2), (2, 3)], 1.)
    events, states = fixture()
    bad = copy.deepcopy(states[0])
    bad['index'] = 99
    with pytest.raises(ValueError, match='prefix keys'):
        prepare(events, states+[bad], [(1, 2), (2, 3)], 1.)


def test_raw_motion_and_observation_not_replaced_by_alignment():
    events, states = fixture()
    events.append(dict(kind='pose', robot=2, pose_index=1, timestamp_s=1.,
        support_max_time_s=1., delta=[.3, .1, .2], duration_s=1., from_pose_index=0))
    states.append(dict(kind='pose', robot=2, index=1, timestamp_s=1., x=.3, y=.1, yaw=.2))
    result = prepare(events, states, [(1, 2), (2, 3)], 1.)
    robot = result['robots'][1]
    assert robot['odometry'][0]['delta'] == [.3, .1, .2]
    assert robot['poses'][1]['value'] == pytest.approx([-9.7, -1.9, .2])
    assert robot['observations'][0]['range'] == 1.
