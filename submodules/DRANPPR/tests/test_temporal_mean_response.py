import copy
import json
from pathlib import Path

import numpy as np
import pytest


def test_linear_response_equals_full_augmented_least_squares_and_covariance():
    from experiments.mean_response import fit_response, predict_response
    rng = np.random.default_rng(84)
    a, b, directions = rng.normal(size=(40, 8)), rng.normal(size=40), rng.normal(size=(40, 3))
    fit = fit_response(a, b, directions)
    augmented = np.column_stack([a, directions])
    solution = np.linalg.lstsq(augmented, -b, rcond=None)[0]
    np.testing.assert_allclose(fit['mean'], solution[8:], atol=1e-12)
    np.testing.assert_allclose(fit['step']+fit['sensitivity']@fit['mean'], solution[:8], atol=1e-12)
    j, d = rng.normal(size=(2, 8)), rng.normal(size=(2, 3))
    shift, extra = predict_response(j, d, fit)
    np.testing.assert_allclose(shift, np.column_stack([j, d])@solution, atol=1e-12)
    base = j@np.linalg.inv(a.T@a)@j.T
    expected = np.column_stack([j, d])@np.linalg.inv(augmented.T@augmented)@np.column_stack([j, d]).T
    np.testing.assert_allclose(base+extra, expected, atol=1e-12)
    assert fit['orthogonality_relative'] < 1e-12


def test_rank_deficient_state_or_mean_is_reported_without_a_prior():
    from experiments.mean_response import fit_response
    rng = np.random.default_rng(85)
    a, b = rng.normal(size=(30, 5)), rng.normal(size=30)
    with pytest.raises(ValueError, match='mean'):
        fit_response(a, b, np.column_stack([a[:, 0], rng.normal(size=30)]))
    with pytest.raises(ValueError, match='state'):
        fit_response(np.column_stack([a, a[:, 0]]), b, rng.normal(size=(30, 2)))


def test_range_response_uses_physical_units_and_original_column_order():
    from experiments.mean_response import fit_native_response
    from experiments.network_noise_prediction import sparse_model
    root = Path(__file__).resolve().parents[1]
    source = root/'results/dranppr_network_noise_transfer_qr_20261003/d4/training/linearization.json'
    data = json.loads(source.read_text())
    fit = fit_native_response(data)
    a, b, columns = sparse_model(data)
    assert fit['columns'] == columns
    assert fit['robots'] == [1, 2, 3, 4, 5]
    assert fit['mean'].shape == (5,)
    assert np.linalg.eigvalsh(fit['covariance']).min() > 0
    assert np.linalg.norm(a.T@(b+a@fit['step'])) < 1e-8


def archived_forecast():
    from experiments.evaluate_incremental_reference import read_rows
    root = Path(__file__).resolve().parents[1]
    folder = root/'results/dranppr_network_noise_transfer_qr_20261003/d4'
    package = json.loads((folder/'t30_nominal.json').read_text())
    linear = json.loads((folder/'t30_nominal/linearization.json').read_text())
    states = read_rows(folder/'t30_nominal/final_estimates.csv')
    saved = json.loads((folder/'t30_nominal/forecast.json').read_text())
    events = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    return package, linear, states, events, saved


def test_zero_response_preserves_archive_and_uncertainty_never_shrinks():
    from experiments.mean_response import fit_native_response
    from experiments.network_noise_prediction import forecast
    package, linear, states, events, saved = archived_forecast()
    fit = fit_native_response(linear)
    zero = dict(fit, step=np.zeros_like(fit['step']), mean=np.zeros(5), covariance=np.zeros((5, 5)))
    one = forecast(package, states, linear, events, saved['checkpoint_s'], 5., np.ones(5))
    two = forecast(package, states, linear, events, saved['checkpoint_s'], 5., np.ones(5), mean_response=zero)
    for p, q, reference in zip(one['predictions'], two['predictions'], saved['predictions']):
        np.testing.assert_allclose(p['mean'], reference['mean'], atol=1e-12)
        np.testing.assert_allclose(p['covariance'], reference['covariance'], atol=1e-12)
        np.testing.assert_array_equal(p['mean'], q['mean'])
        np.testing.assert_array_equal(p['covariance'], q['covariance'])
    full = forecast(package, states, linear, events, saved['checkpoint_s'], 5., np.ones(5), mean_response=fit)
    for p, q in zip(one['predictions'], full['predictions']):
        assert np.linalg.eigvalsh(np.array(q['covariance'])-p['covariance']).min() > -1e-12


def test_future_observation_values_cannot_change_response_forecast():
    from experiments.mean_response import fit_native_response
    from experiments.network_noise_prediction import forecast
    package, linear, states, events, saved = archived_forecast()
    fit = fit_native_response(linear)
    changed = copy.deepcopy(events)
    for e in changed:
        if e['kind'] == 'observation' and e['timestamp_s'] > saved['checkpoint_s']:
            e['range_m'] += 10.
            e['bearing_rad'] += .2
    args = (package, states, linear)
    one = forecast(*args, events, saved['checkpoint_s'], 5., np.ones(5), mean_response=fit)
    two = forecast(*args, changed, saved['checkpoint_s'], 5., np.ones(5), mean_response=fit)
    assert len(one['predictions']) == len(two['predictions']) > 0
    for p, q in zip(one['predictions'], two['predictions']):
        assert p['mean'] == q['mean'] and p['covariance'] == q['covariance']
        assert p['nll'] != q['nll']


def test_retracted_means_compose_body_steps_and_do_not_use_future_values():
    from experiments.mean_response import retracted_means
    package = dict(robots=[dict(id=1, poses=[dict(index=1, timestamp_s=2.)],
                              objects=[dict(label=7, cluster=0)])])
    states = [dict(robot=1, kind='pose', index=1, x=1., y=0., yaw=np.pi/2),
              dict(robot=1, kind='landmark', cluster=0, x=2., y=2.)]
    response = dict(columns={('x', 1, 1): [0, 1, 2], ('l', -1, 0): [3, 4]},
        step=np.array([.2, .1, .05, .1, -.1]), sensitivity=np.zeros((5, 1)),
        mean=np.array([.3]), robots=[1])
    events = [dict(kind='pose', robot=1, timestamp_s=3., pose_index=2, from_pose_index=1,
                   available_time_s=3., support_max_time_s=3., delta=[.1, 0., 0.], duration_s=1.),
              dict(kind='observation', robot=1, timestamp_s=3., pose_index=2,
                   available_time_s=3., support_max_time_s=3., landmark_subject=7, source_row=8,
                   range_m=20., bearing_rad=-2.)]
    one = retracted_means(package, states, events, 2., 5., response)
    dx, dy = 2.1-(.9-.1*np.sin(.05)), 1.9-(.2+.1*np.cos(.05))
    np.testing.assert_allclose(one[0]['mean'],
        [np.arctan2(dy, dx)-np.pi/2-.05, np.hypot(dx, dy)+.3], atol=1e-12)
    changed = copy.deepcopy(events)
    changed[-1].update(range_m=100., bearing_rad=1.)
    assert retracted_means(package, states, changed, 2., 5., response) == one
