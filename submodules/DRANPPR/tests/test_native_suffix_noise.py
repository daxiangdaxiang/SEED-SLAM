import json
import math

import pytest

from test_native_conditional_replay import rows, run
from test_native_joint_audit import package
from test_native_matched_online import stream


FLAG = '--suffix-added-bearing-variance'


@pytest.mark.parametrize('mode', ['joint_cached_one', 'joint_conditional_cached_one'])
def test_added_variance_changes_only_suffix_bearing_precision(tmp_path, mode):
    data = package()
    events = stream(data)
    observation = next(e for e in events if e['timestamp_s'] == 3. and
                       e['robot'] == 1 and e['kind'] == 'observation')
    observation.update(landmark_subject=6, bearing_rad=.2, range_m=3.)
    outputs = []
    for name, extra in [('nominal', ()), ('added', (FLAG, '.0225'))]:
        p, out = run(tmp_path/name, mode, data=data, events=events,
                     lazy=True, extra=extra)
        assert p.returncode == 0, p.stderr
        outputs.append(out)
    a, b = [json.loads((p/'summary.json').read_text()) for p in outputs]
    assert b['suffix_added_bearing_variance'] == .0225
    assert b['suffix_bearing_sigma'] == pytest.approx(math.sqrt(.025))
    assert b['noise_parameters_learned'] is False
    assert b['initial_cost'] == a['initial_cost']
    assert b['prefix_physical_factors'] == a['prefix_physical_factors'] == 22
    assert b['physical_factors'] == a['physical_factors']
    assert b['root_displacement'] == 0.
    assert b['consensus_max_m'] < 1e-12
    # First insertion uses the unchanged prefix pose and existing landmark.
    x, y, yaw = data['robots'][0]['poses'][-1]['value']
    x, y = x+.4*math.cos(yaw), y+.4*math.sin(yaw)
    ox, oy = data['robots'][0]['objects'][0]['point']
    error = math.atan2(oy-y, ox-x)-yaw-observation['bearing_rad']
    error = math.atan2(math.sin(error), math.cos(error))
    expected = .5*error**2*(1/.05**2-1/(.05**2+.0225))
    before = [float(rows(p/'updates.csv')[0]['cost_before']) for p in outputs]
    assert before[0]-before[1] == pytest.approx(expected, abs=1e-8)


def test_explicit_zero_is_bitwise_default_and_nonzero_is_causal(tmp_path):
    p, default = run(tmp_path/'default', 'joint_cached_one', lazy=True)
    assert p.returncode == 0, p.stderr
    p, zero = run(tmp_path/'zero', 'joint_cached_one', lazy=True, extra=(FLAG, '0'))
    assert p.returncode == 0, p.stderr
    for name in ['final_estimates.csv', 'causal_estimates.csv', 'control_messages.jsonl']:
        assert (default/name).read_bytes() == (zero/name).read_bytes()
    p, short = run(tmp_path/'short', 'joint_cached_one', duration=3., lazy=True,
                   extra=(FLAG, '.0225'))
    assert p.returncode == 0, p.stderr
    events = stream(package())
    for e in events:
        if e['timestamp_s'] > 3. and e['kind'] == 'observation':
            e['bearing_rad'] += .8
    p, long = run(tmp_path/'long', 'joint_cached_one', events=events, lazy=True,
                  extra=(FLAG, '.0225'))
    assert p.returncode == 0, p.stderr
    earlier = (short/'causal_estimates.csv').read_text().splitlines()
    assert earlier == (long/'causal_estimates.csv').read_text().splitlines()[:len(earlier)]


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf', '0.0225junk'])
def test_invalid_variance_is_rejected(tmp_path, value):
    p, _ = run(tmp_path/'bad', 'joint_cached_one', extra=(FLAG, value))
    assert p.returncode != 0
    assert 'invalid suffix bearing variance' in p.stderr


@pytest.mark.parametrize('extra,warm,message', [
    ((FLAG, '.0225', FLAG, '.0225'), True, 'duplicate suffix bearing variance'),
    ((FLAG, '.0225'), False, 'suffix bearing variance requires common-past')])
def test_variance_contract_requires_single_option_and_common_past(tmp_path, extra, warm, message):
    p, _ = run(tmp_path/'bad', 'joint_cached_one', warm=warm, extra=extra)
    assert p.returncode != 0
    assert message in p.stderr
