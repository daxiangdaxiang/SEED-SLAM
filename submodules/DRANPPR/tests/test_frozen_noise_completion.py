import copy
import json
import math

import pytest

from test_native_joint_audit import package, run as frozen_run
from test_native_conditional_replay import run as replay_run, rows
from test_native_matched_online import stream


def weighted():
    data = package()
    data['schema'] = 'dranppr_frozen_joint_v2'
    for robot in data['robots']:
        for factor in robot['odometry']:
            factor['sigmas'] = [s*math.sqrt(factor['dt']) for s in [.1, .1, .05]]
        for factor in robot['observations']:
            factor['sigmas'] = [.05, .1]
    return data


def test_weighted_frozen_loss_uses_explicit_covariance(tmp_path):
    process, original = frozen_run(tmp_path, package(), 'central_lm', 1)
    assert process.returncode == 0, process.stderr
    data = weighted()
    for robot in data['robots']:
        for factor in robot['odometry']+robot['observations']:
            factor['sigmas'] = [2*s for s in factor['sigmas']]
    other = tmp_path/'weighted'
    other.mkdir()
    process, output = frozen_run(other, data, 'central_lm', 1)
    assert process.returncode == 0, process.stderr
    a, b = [json.loads((p/'summary.json').read_text()) for p in [original, output]]
    assert b['initial_cost'] == pytest.approx(a['initial_cost']/4, rel=1e-12)


def test_only_one_bearing_precision_changes(tmp_path):
    data = weighted()
    process, original = frozen_run(tmp_path, data, 'central_gn', 1)
    assert process.returncode == 0, process.stderr
    robot = data['robots'][0]
    factor = robot['observations'][0]
    x, y, yaw = robot['poses'][factor['pose']]['value']
    ox, oy = robot['objects'][factor['cluster']]['point']
    error = math.atan2(oy-y, ox-x)-yaw-factor['bearing']
    error = math.atan2(math.sin(error), math.cos(error))
    reduction = .5*error**2*(1/.05**2-1/.1**2)
    factor['sigmas'][0] = .1
    other = tmp_path/'one'
    other.mkdir()
    process, output = frozen_run(other, data, 'central_gn', 1)
    assert process.returncode == 0, process.stderr
    a, b = [json.loads((p/'summary.json').read_text()) for p in [original, output]]
    assert a['initial_cost']-b['initial_cost'] == pytest.approx(reduction, abs=1e-12)


@pytest.mark.parametrize('mutation', ['missing', 'zero', 'negative', 'dimension', 'string', 'v1'])
def test_invalid_covariance_rejected(tmp_path, mutation):
    data = weighted()
    factor = data['robots'][0]['observations'][0]
    if mutation == 'missing':
        del factor['sigmas']
    elif mutation == 'v1':
        data['schema'] = 'dranppr_frozen_joint_v1'
    else:
        factor['sigmas'] = {'zero': [0., .1], 'negative': [-.1, .1],
                            'dimension': [.1], 'string': ['.1', .1]}[mutation]
    process, _ = frozen_run(tmp_path, data, 'central_lm', 1)
    assert process.returncode != 0


def test_weighted_frozen_graph_cannot_silently_reset_in_replay(tmp_path):
    process, _ = replay_run(tmp_path/'online', 'joint_cached_one', data=weighted())
    assert process.returncode != 0
    assert 'schema' in process.stderr.lower()


@pytest.mark.parametrize('variance', [0., .0225])
def test_terminal_export_preserves_exact_physical_loss(tmp_path, variance):
    from experiments.prepare_frozen_completion import prepare
    data = package()
    events = stream(data)
    process, online = replay_run(tmp_path/'online', 'joint_cached_one', data=data, events=events,
        lazy=True, extra=['--suffix-added-bearing-variance', str(variance)])
    assert process.returncode == 0, process.stderr
    states = rows(online/'final_estimates.csv')
    frozen = prepare(events, states, data, 4., variance)
    process, out = frozen_run(tmp_path, frozen, 'central_lm', 1)
    assert process.returncode == 0, process.stderr
    a, b = [json.loads((p/'summary.json').read_text()) for p in [online, out]]
    assert b['initial_cost'] == pytest.approx(a['physical_cost'], rel=1e-10, abs=1e-9)
    assert sum(len(r['odometry'])+len(r['observations']) for r in frozen['robots']) == a['physical_factors']
    for r in frozen['robots']:
        for o in r['observations']:
            suffix = r['poses'][o['pose']]['timestamp_s'] > 2.
            assert o['sigmas'][0] == math.sqrt(.05**2+(variance if suffix else 0.))
    bad = copy.deepcopy(states)
    next(r for r in bad if r['kind'] == 'landmark')['x'] = '999'
    with pytest.raises(ValueError, match='shared'):
        prepare(events, bad, data, 4., variance)
    with pytest.raises(ValueError, match='state'):
        prepare(events, states[:-1], data, 4., variance)
    bad_events = copy.deepcopy(events)
    next(e for e in bad_events if e['kind'] == 'observation')['range_m'] += .1
    with pytest.raises(ValueError, match='prefix'):
        prepare(bad_events, states, data, 4., variance)


def test_predeclared_grid_and_state_comparison():
    from experiments.audit_frozen_noise_completion import grid, state_difference
    assert len(grid()) == len(set(grid())) == 26
    assert sum(r[3] == 'routed' for r in grid()) == 2
    left = [dict(robot=1, kind='pose', index=0, x=1., y=2., yaw=.1)]
    right = [dict(left[0], yaw=.1+2*math.pi, x=4., y=6.)]
    diff = state_difference(left, right)
    assert diff['max_xy_m'] == 5.
    assert diff['max_yaw_rad'] < 1e-14
    with pytest.raises(ValueError, match='identity'):
        state_difference(left, left*2)


def test_reference_residuals_are_posthoc_and_angle_wrapped():
    import numpy as np
    from benchmarks.mrclam.evaluate import GroundTruthData, RobotGroundTruthSeries
    from experiments.analyze_frozen_noise_completion import reference_residuals
    truth = GroundTruthData(robots={1:RobotGroundTruthSeries(1, np.array([0., 2.]),
        np.array([[0., 0.], [0., 0.]]), np.zeros(2))}, landmarks={6:np.array([-1., 0.])})
    event = dict(robot=1, kind='observation', timestamp_s=1., landmark_subject=6,
                 range_m=1.2, bearing_rad=-math.pi+.1)
    values = reference_residuals([event], truth, {}, .5, 2.)
    assert values['suffix']['bearing']['mean'] == pytest.approx(.1)
    assert values['suffix']['range']['mean'] == pytest.approx(.2)
    assert values['prefix']['bearing']['count'] == 0
    assert values['unsupported'] == 0
