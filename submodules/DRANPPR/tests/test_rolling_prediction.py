import copy

import numpy as np
import pytest


def test_rolling_windows_cover_partial_ends_without_overlap():
    from experiments.audit_rolling_prediction import windows
    pairs = windows(15.534, 32., 5.)
    assert pairs == [(15.534, 20.), (20., 25.), (25., 30.), (30., 32.)]
    for t in [15.535, 20., 20.001, 25., 30., 32.]:
        assert sum(a < t <= b for a, b in pairs) == 1
    assert windows(20., 30., 5.) == [(20., 25.), (25., 30.)]


@pytest.mark.parametrize('start,end,step', [(20, 20, 5), (30, 20, 5),
                                          (0, 20, 0), (float('nan'), 20, 5)])
def test_rolling_windows_reject_invalid_schedule(start, end, step):
    from experiments.audit_rolling_prediction import windows
    with pytest.raises(ValueError):
        windows(start, end, step)


def support_fixture():
    package = dict(robots=[dict(id=1, objects=[dict(label=6)]),
                           dict(id=2, objects=[dict(label=7)])])
    events = [dict(kind='observation', robot=1, source_row=i, timestamp_s=t,
                   available_time_s=t, support_max_time_s=t,
                   landmark_subject=s) for i, t, s in
              [(0, 10., 6), (1, 11., 6), (2, 12., 7), (3, 15., 8), (4, 16., 6)]]
    rows = [dict(robot=1, source_row=1, timestamp_s=11., subject=6)]
    return package, events, rows


def test_support_ledger_retains_every_unsupported_event_and_reason():
    from experiments.audit_rolling_prediction import support_ledger
    package, events, rows = support_fixture()
    ledger = support_ledger(package, events, 10., 15., rows)
    assert [r['source_row'] for r in ledger] == [1, 2, 3]
    assert [r['status'] for r in ledger] == [
        'scored', 'no_local_object_copy', 'unseen_object']
    assert events[0]['timestamp_s'] == 10.


def test_support_ledger_rejects_missing_duplicate_and_spurious_predictions():
    from experiments.audit_rolling_prediction import support_ledger
    package, events, rows = support_fixture()
    for changed in [[], rows+rows, rows+[dict(rows[0], source_row=2, subject=7)]]:
        with pytest.raises(ValueError, match='support'):
            support_ledger(package, events, 10., 15., changed)
    broken = copy.deepcopy(events)
    broken[1]['support_max_time_s'] = 12.
    with pytest.raises(ValueError, match='support'):
        support_ledger(package, broken, 10., 15., rows)


def test_balanced_scoring_does_not_count_observations_as_independent_frames():
    from experiments.audit_rolling_prediction import statistics
    rows = [dict(robot=1, timestamp_s=t, window=w, nll=n, energy=2*n,
                 innovation=[0., n], covariance=np.eye(2).tolist())
            for t, w, n in [(1., 0, 1.), (1., 0, 1.), (2., 1, 4.)]]
    result = statistics(rows)
    assert result['count'] == 3
    assert result['frame_count'] == 2
    assert result['mean_nll'] == 2.
    assert result['frame_equal_nll'] == 2.5
    assert result['window_equal_nll'] == 2.5
    assert result['range_rmse_m'] == pytest.approx(np.sqrt(6.))
    assert result['coverage_95'] == pytest.approx(2/3)
    assert statistics([]) == dict(count=0, frame_count=0)


def test_independent_weight_check_detects_prefix_or_measurement_changes():
    from experiments.analyze_rolling_prediction import check_weights
    nominal = dict(schema='dranppr_frozen_joint_v2', robots=[dict(id=1, poses=[dict(index=0, timestamp_s=0.),
        dict(index=1, timestamp_s=1.)], objects=[], odometry=[dict(source=0,
        target=1, dt=1., delta=[1., 0., 0.], sigmas=[.1, .1, .05])],
        observations=[dict(pose=1, range=2., bearing=.5, sigmas=[.05, .1])])])
    scales = [4., 9., 16., 25., 36.]
    adapted = copy.deepcopy(nominal)
    adapted['robots'][0]['odometry'][0]['sigmas'] = [.2, .3, .2]
    adapted['robots'][0]['observations'][0]['sigmas'] = [.25, .6]
    assert check_weights(nominal, adapted, scales, .5) < 1e-14
    with pytest.raises(ValueError, match='weight'):
        check_weights(nominal, adapted, scales, 1.)
    adapted['robots'][0]['observations'][0]['range'] = 3.
    with pytest.raises(ValueError, match='input'):
        check_weights(nominal, adapted, scales, .5)


def test_independent_weight_check_supports_implicit_nominal_v1_prefix():
    from experiments.analyze_rolling_prediction import check_weights
    package = dict(schema='dranppr_frozen_joint_v1', robots=[dict(id=1,
        poses=[dict(index=0, timestamp_s=0.), dict(index=1, timestamp_s=1.)],
        objects=[], odometry=[dict(source=0, target=1, dt=1., delta=[1., 0., 0.])],
        observations=[dict(pose=1, range=2., bearing=.5)])])
    assert check_weights(package, package, [4.]*5, 1.) == 0.
    invalid = copy.deepcopy(package)
    invalid['schema'] = 'dranppr_frozen_joint_v2'
    with pytest.raises(ValueError, match='schema'):
        check_weights(invalid, invalid, [1.]*5, 1.)


def test_independent_prediction_check_reconstructs_measurement_and_score():
    from experiments.analyze_rolling_prediction import check_prediction
    row = dict(mean=[3., 1.], innovation=[.5, 2.], covariance=[[.25, 0.], [0., 4.]],
               energy=2., nll=1.+np.log(2*np.pi))
    event = dict(bearing_rad=3.5-2*np.pi, range_m=3.)
    assert max(check_prediction(row, event).values()) < 1e-14
    with pytest.raises(ValueError, match='score'):
        check_prediction(dict(row, nll=10.), event)
