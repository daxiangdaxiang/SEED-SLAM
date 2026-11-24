import math

import pytest

from experiments.audit_native_interface_support import analyze


def fixture():
    events, states = [], []
    for time, robots, index, landmark in [(0., [1, 2], 0, 6), (1., [1], 1, 7), (2., [2], 1, 7)]:
        for robot in robots:
            events.append(dict(kind='pose', robot=robot, pose_index=index, timestamp_s=time,
                               available_time_s=time, support_max_time_s=time))
            events.append(dict(kind='observation', robot=robot, pose_index=index, timestamp_s=time,
                               support_max_time_s=time, landmark_subject=landmark, source_row=index))
            states.append(dict(kind='pose', robot=robot, index=index, timestamp_s=time, x=0., y=0., yaw=0.))
    for robot, points in [(1, [(0., 0.), (1., 0.)]), (2, [(10., 2.), (10., 3.)])]:
        for index, point in enumerate(points):
            states.append(dict(kind='landmark', robot=robot, index=6+index, x=point[0], y=point[1]))
    return events, states


def test_arrival_support_and_frozen_registration_are_separate():
    events, states = fixture()
    result = analyze(events, states, [(1, 2)], 2.)
    assert result['truth_used'] is False
    edge = result['edges'][0]
    assert edge['first_two_shared_elapsed_s'] == 2.
    assert edge['common_landmarks'] == [6, 7]
    assert edge['registration']['status'] == 'estimated'
    assert edge['registration']['rms_m'] < 1e-12
    assert edge['registration']['target_from_source'] == pytest.approx([-2., 10., -math.pi/2])


def test_no_future_observation_or_state_enters_prefix():
    events, states = fixture()
    prefix = [s for s in states if not (s['robot'] == 2 and (s['index'] == 1 or s['index'] == 7))]
    result = analyze(events, prefix, [(1, 2)], 1.)
    assert result['edges'][0]['first_two_shared_elapsed_s'] is None
    assert result['edges'][0]['registration']['status'] == 'insufficient_correspondences'
    with pytest.raises(ValueError, match='prefix keys'):
        analyze(events, states, [(1, 2)], 1.)


def test_noncausal_and_duplicate_snapshots_rejected():
    events, states = fixture()
    events[1]['support_max_time_s'] = 9.
    with pytest.raises(ValueError, match='noncausal'):
        analyze(events, states, [(1, 2)], 2.)
    events, states = fixture()
    with pytest.raises(ValueError, match='duplicate state'):
        analyze(events, states+[states[0]], [(1, 2)], 2.)
    states[0]['timestamp_s'] = .1
    with pytest.raises(ValueError, match='timestamp'):
        analyze(events, states, [(1, 2)], 2.)


def test_degenerate_geometry_not_hidden_by_two_labels():
    events, states = fixture()
    for state in states:
        if state['kind'] == 'landmark' and state['robot'] == 1:
            state['x'] = state['y'] = 0.
    result = analyze(events, states, [(1, 2)], 2.)
    assert result['edges'][0]['first_two_shared_elapsed_s'] == 2.
    assert result['edges'][0]['registration']['status'] == 'degenerate_geometry'


def test_invalid_topology_rejected():
    events, states = fixture()
    for links in [[(1, 1)], [(1, 2), (2, 1)], []]:
        with pytest.raises(ValueError, match='topology'):
            analyze(events, states, links, 2.)
