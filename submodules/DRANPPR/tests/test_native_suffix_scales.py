import json
import math

import pytest

from test_native_conditional_replay import rows, run
from test_native_joint_audit import package
from test_native_matched_online import stream

FLAG = '--suffix-variance-multipliers'


def test_suffix_scales_preserve_prefix_and_match_physical_insertion_loss(tmp_path):
    data = package()
    events = stream(data)
    obs = next(e for e in events if e['robot']==1 and e['timestamp_s']==3. and e['kind']=='observation')
    obs.update(landmark_subject=6, bearing_rad=.2, range_m=3.)
    outputs = []
    for name, extra in [('nominal', ()), ('scaled', (FLAG, '4', '9', '16', '25', '36'))]:
        p, out = run(tmp_path/name, 'joint_cached_one', data=data, events=events, lazy=True, extra=extra)
        assert p.returncode == 0, p.stderr
        outputs.append(out)
    a, b = [json.loads((p/'summary.json').read_text()) for p in outputs]
    assert b['suffix_variance_multipliers'] == [4, 9, 16, 25, 36]
    assert b['initial_cost'] == a['initial_cost']
    assert b['prefix_physical_factors'] == a['prefix_physical_factors']
    x, y, yaw = data['robots'][0]['poses'][-1]['value']
    x, y = x+.4*math.cos(yaw), y+.4*math.sin(yaw)
    ox, oy = data['robots'][0]['objects'][0]['point']
    bearing = math.atan2(oy-y, ox-x)-yaw-obs['bearing_rad']
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    distance = math.hypot(ox-x, oy-y)-obs['range_m']
    expected = .5*(bearing**2/.05**2*(1-1/25)+distance**2/.1**2*(1-1/36))
    before = [float(rows(p/'updates.csv')[0]['cost_before']) for p in outputs]
    assert before[0]-before[1] == pytest.approx(expected, abs=1e-8)


def test_explicit_unit_scales_are_bitwise_default_and_scaled_run_is_causal(tmp_path):
    p, original = run(tmp_path/'original', 'joint_cached_one', lazy=True)
    assert p.returncode == 0, p.stderr
    p, unit = run(tmp_path/'unit', 'joint_cached_one', lazy=True, extra=(FLAG, *(['1']*5)))
    assert p.returncode == 0, p.stderr
    for name in ['final_estimates.csv', 'causal_estimates.csv', 'control_messages.jsonl']:
        assert (original/name).read_bytes() == (unit/name).read_bytes()
    extra = (FLAG, '.2', '.3', '.8', '.02', '.7')
    p, short = run(tmp_path/'short', 'joint_cached_one', duration=3., lazy=True, extra=extra)
    assert p.returncode == 0, p.stderr
    events = stream(package())
    for e in events:
        if e['kind']=='observation' and e['timestamp_s']>3.:
            e['range_m'] += 2.
    p, long = run(tmp_path/'long', 'joint_cached_one', events=events, lazy=True, extra=extra)
    assert p.returncode == 0, p.stderr
    earlier = (short/'causal_estimates.csv').read_text().splitlines()
    assert earlier == (long/'causal_estimates.csv').read_text().splitlines()[:len(earlier)]


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '.5junk'])
def test_suffix_scales_reject_invalid_values(tmp_path, value):
    p, _ = run(tmp_path/'bad', 'joint_cached_one', extra=(FLAG, value, '1', '1', '1', '1'))
    assert p.returncode != 0
    assert 'invalid suffix variance multipliers' in p.stderr


@pytest.mark.parametrize('extra,warm,message', [
    ((FLAG, *(['1']*5), FLAG, *(['1']*5)), True, 'duplicate suffix variance multipliers'),
    ((FLAG, *(['1']*5)), False, 'suffix variance multipliers require common-past')])
def test_suffix_scales_require_one_flag_and_common_past(tmp_path, extra, warm, message):
    p, _ = run(tmp_path/'bad', 'joint_cached_one', warm=warm, extra=extra)
    assert p.returncode != 0
    assert message in p.stderr
