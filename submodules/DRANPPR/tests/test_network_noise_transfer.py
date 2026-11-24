import copy
import json
import math

import numpy as np
import pytest

from test_network_noise_support import exported
from test_native_joint_audit import package
from experiments.evaluate_incremental_reference import read_rows


def test_sparse_marginals_match_native_dense_information(tmp_path):
    from experiments.network_noise_prediction import sparse_model, selected_covariance
    from experiments.network_noise_support import assemble
    out = exported(tmp_path)
    data = json.loads((out/'linearization.json').read_text())
    a, b, _ = assemble(data)
    sparse, rhs, columns = sparse_model(data)
    assert np.array_equal(sparse.toarray(), a)
    assert np.array_equal(rhs, b)
    chosen = list(columns)[-3:]
    covariance, selection, residual = selected_covariance(sparse, columns, chosen)
    indices = [i for key in chosen for i in columns[key]]
    assert np.allclose(covariance, np.linalg.inv(a.T@a)[np.ix_(indices, indices)], atol=1e-11)
    assert residual < 1e-10
    assert len(selection) == 3


def test_selected_covariance_avoids_normal_equation_precision_loss():
    from scipy.sparse import csc_matrix
    from experiments.network_noise_prediction import selected_covariance
    a = np.array([[1.,1.],[0.,1e-6]])
    columns = {('l',-1,0): [0,1]}
    covariance, _, residual = selected_covariance(csc_matrix(a), columns, list(columns))
    inverse = np.linalg.inv(a)
    expected = inverse@inverse.T
    assert np.linalg.norm(covariance-expected)/np.linalg.norm(expected) < 1e-12
    assert residual < 1e-10


def test_polar_jacobian_matches_native_export(tmp_path):
    from experiments.network_noise_prediction import polar_model
    out = exported(tmp_path)
    rows = read_rows(out/'final_estimates.csv')
    poses = {(int(r['robot']), int(r['index'])): [float(r[k]) for k in ['x','y','yaw']]
             for r in rows if r['kind'] == 'pose'}
    points = {int(r['cluster']): [float(r[k]) for k in ['x','y']]
              for r in rows if r['kind'] == 'landmark'}
    factors = json.loads((out/'linearization.json').read_text())['factors']
    for f in factors:
        if f['kind'] != 'observation' or len(f['terms']) != 2:
            continue
        pose = next(t for t in f['terms'] if t['kind'] == 'x')
        point = next(t for t in f['terms'] if t['kind'] == 'l')
        _, hp, hl = polar_model(poses[f['robot'], pose['index']], points[point['index']])
        assert np.allclose(hp/np.array([.05,.1])[:,None], pose['a'], atol=1e-11)
        assert np.allclose(hl/np.array([.05,.1])[:,None], point['a'], atol=1e-11)


def test_forecast_propagation_includes_body_rotation_and_process_noise():
    from experiments.network_noise_prediction import propagate
    pose, f, q = propagate(np.array([2.,3.,math.pi/2]), np.eye(3), np.zeros((3,3)),
                          [1.,0.,math.pi/2], 2., np.array([2.,3.,4.]))
    assert np.allclose(pose, [2.,4.,math.pi])
    assert np.allclose(f, [[0,1,1],[-1,0,0],[0,0,1]], atol=1e-14)
    assert np.allclose(q, np.diag([.04,.06,.02]))
    _, f2, q2 = propagate(pose, f, q, [0.,1.,0.], 1., np.ones(3))
    step = np.array([[1.,0.,-1.],[0.,1.,0.],[0.,0.,1.]])
    assert np.allclose(f2, step@f)
    assert np.allclose(q2, step@q@step.T+np.diag([.01,.01,.0025]))


def forecast_fixture(tmp_path):
    out = exported(tmp_path)
    data = package()
    events = [dict(kind='pose', robot=1, pose_index=3, from_pose_index=2,
                   timestamp_s=3., available_time_s=3., support_max_time_s=3.,
                   delta=[.4,0.,0.], duration_s=1.),
              dict(kind='observation', robot=1, pose_index=3, timestamp_s=3.,
                   available_time_s=3., support_max_time_s=3., source_row=100,
                   landmark_subject=6, range_m=3., bearing_rad=1.)]
    return data, read_rows(out/'final_estimates.csv'), json.loads((out/'linearization.json').read_text()), events


def test_forecast_does_not_assimilate_any_future_observation(tmp_path):
    from experiments.network_noise_prediction import forecast
    data, states, linear, events = forecast_fixture(tmp_path)
    before = copy.deepcopy(events)
    one = forecast(data, states, linear, events, 2., 5., np.ones(5))
    changed = copy.deepcopy(events)
    changed[-1]['range_m'] += 10.
    changed[-1]['bearing_rad'] += .2
    events.append(dict(events[-1], source_row=101, landmark_subject=7))
    changed.append(dict(events[-1]))
    before = copy.deepcopy(events)
    one = forecast(data, states, linear, events, 2., 5., np.ones(5))
    two = forecast(data, states, linear, changed, 2., 5., np.ones(5))
    assert events == before
    p, q = one['predictions'][0], two['predictions'][0]
    assert p['mean'] == q['mean'] and p['covariance'] == q['covariance']
    assert p['nll'] != q['nll']
    assert one['predictions'][1] == two['predictions'][1]
    assert p['observation_covariance'] == [[.0025,0.],[0.,.01]] or np.allclose(p['observation_covariance'],np.diag([.0025,.01]))
    extended = events+[dict(events[-1], source_row=102, landmark_subject=999)]
    assert forecast(data, states, linear, extended, 2., 5., np.ones(5))['unknown_object_observations'] == 1
    invalid = copy.deepcopy(events)
    invalid[0]['support_max_time_s'] = 10.
    with pytest.raises(ValueError, match='support'):
        forecast(data, states, linear, invalid, 2., 5., np.ones(5))


def test_forecast_covariance_is_not_the_postfit_residual_covariance(tmp_path):
    from experiments.network_noise_prediction import forecast
    data, states, linear, events = forecast_fixture(tmp_path)
    result = forecast(data, states, linear, events, 2., 5., np.ones(5))
    p = result['predictions'][0]
    assert np.linalg.eigvalsh(np.array(p['covariance'])-p['observation_covariance']).min() > 0
    innovation = np.asarray(p['innovation'])
    total = np.asarray(p['covariance'])
    energy = innovation@np.linalg.solve(total,innovation)
    assert p['energy'] == pytest.approx(energy)
    assert p['nll'] == pytest.approx(.5*(energy+np.linalg.slogdet(total)[1]+2*np.log(2*np.pi)))


def test_transfer_grid_and_prediction_summary_are_fixed():
    from experiments.audit_network_noise_transfer import grid, prediction_summary
    cases = grid()
    assert len(cases) == 32
    assert set(d for d,t,a in cases) == {1,4}
    assert set(t for d,t,a in cases) == {30,60,90,120}
    assert set(a for d,t,a in cases) == {'nominal','joint_r','joint_qr','split_qr'}
    result = prediction_summary([dict(energy=0., nll=-1.), dict(energy=8., nll=3.)])
    assert result['mean_nll'] == 1. and result['coverage_95'] == .5
    assert result['mean_energy'] == 4. and result['count'] == 2
    with pytest.raises(ValueError):
        prediction_summary([])


def test_predictive_mean_covariance_attribution_preserves_observation_support():
    from experiments.analyze_network_noise_transfer import rescore
    row = dict(robot=1,source_row=9,timestamp_s=3.,subject=6,
               mean=[0.,1.],innovation=[.1,.2],covariance=[[1.,0.],[0.,1.]])
    other = dict(row, mean=[.1,1.2],innovation=[0.,0.],covariance=[[2.,0.],[0.,2.]])
    assert rescore([row],[other])[0]['energy'] == pytest.approx(.025)
    assert rescore([other],[row])[0]['energy'] == 0.
    with pytest.raises(ValueError):
        rescore([row],[dict(other,source_row=10)])
    with pytest.raises(ValueError):
        rescore([row],[dict(other,innovation=[0.,1.])])
