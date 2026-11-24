import math
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.network_noise_prediction import polar_model


def test_error_decomposition_telescopes_and_exposes_cancellation():
    from experiments.audit_predictive_error_sources import decompose, summarize
    result = decompose([.2,5.], [.1,0.,0.], [3.2,4.], [.5,0.,.01],
                       [3.,4.], [.3,0.,0.])
    assert result['reconstruction_max_abs'] < 1e-12
    assert np.allclose(np.sum(list(result['components'].values()),axis=0),result['innovation'])
    assert np.allclose(np.sum(list(result['reversed_components'].values()),axis=0),result['innovation'])
    rows = [dict(components={'measurement_reference':[0.,1.], 'map':[0.,-1.],
                             'motion':[0.,0.], 'checkpoint':[0.,0.]},
                 innovation=[0.,0.], counterfactuals={})]
    stats = summarize(rows)
    assert stats['range_component_cross_moment'][0][1] == -1.
    assert stats['range_reconstructed_mse'] == 0.
    assert stats['components']['measurement_reference']['range']['rmse'] == 1.


def test_circular_decomposition_handles_bearing_branch_crossing():
    from experiments.audit_predictive_error_sources import decompose
    result = decompose([math.pi-.01,2.], [0.,0.,.03], [-2.,-.02],
                       [0.,0.,-.02], [-2.,.02], [0.,0.,.01])
    assert result['reconstruction_max_abs'] < 1e-12
    assert abs(result['innovation'][0]) < .1


def test_pair_distance_variance_matches_finite_differences_and_is_pose_free():
    from experiments.audit_predictive_error_sources import pair_measurement
    q = [.3,2.]
    observations = [dict(bearing_rad=.4,range_m=2.),dict(bearing_rad=-.8,range_m=3.)]
    value, variance = pair_measurement(*observations, q)
    numerical = []
    for index in range(2):
        for key in ['bearing_rad','range_m']:
            left, right = [dict(r) for r in observations], [dict(r) for r in observations]
            left[index][key] += 1e-6
            right[index][key] -= 1e-6
            numerical.append((pair_measurement(*left,q)[0]-pair_measurement(*right,q)[0])/2e-6)
    assert variance == pytest.approx(np.square(numerical)@np.tile([.05**2*q[0],.1**2*q[1]],2),rel=1e-8)
    points = [np.array([r['range_m']*math.cos(r['bearing_rad']),r['range_m']*math.sin(r['bearing_rad'])]) for r in observations]
    moved = [polar_model([1.,-2.,.6],p)[0] for p in points]
    other = [dict(bearing_rad=p[0],range_m=p[1]) for p in moved]
    assert pair_measurement(*other,q)[0] == pytest.approx(value)
    with pytest.raises(ValueError):
        pair_measurement(observations[0],observations[0],q)


def pair_events():
    return [dict(kind='observation',robot=1,timestamp_s=float(t),source_row=2*t+i,
                 available_time_s=float(t),support_max_time_s=float(t),
                 landmark_subject=6+i,bearing_rad=.5*i,range_m=2.+i+.1*t)
            for t in [0,1,2] for i in [0,1]]


def test_pair_evidence_uses_previous_frame_and_preserves_support():
    from experiments.audit_predictive_error_sources import pair_evidence
    events = pair_events()
    support = {(1,2),(1,3),(1,4),(1,5)}
    result = pair_evidence(events,support,[.3,2.],.5)
    assert len(result['pairs']) == 2
    assert result['pairs'][0]['previous_timestamp_s'] == 0.
    assert result['pairs'][1]['previous_timestamp_s'] == 1.
    assert result['supported_observations'] == 4
    changed = [dict(e) for e in events]
    changed[-1]['range_m'] += 1.
    assert pair_evidence(changed,support,[.3,2.],.5)['pairs'][0] == result['pairs'][0]
    assert pair_evidence(events,{(1,2)},[.3,2.],.5)['pairs'] == []
    with pytest.raises(ValueError,match='order'):
        pair_evidence(events[::-1],support,[.3,2.],.5)


def test_window_reconstructs_archived_forecast_before_reference_substitution():
    from benchmarks.mrclam.evaluate import RobotGroundTruthSeries
    from experiments.audit_predictive_error_sources import audit_window
    package = dict(root_robot=1,robots=[dict(id=1,poses=[dict(index=0,timestamp_s=0.,value=[.1,0.,0.])],
                                           objects=[dict(cluster=0,label=6,point=[3.2,4.])])])
    states = [dict(robot=1,kind='pose',index=0,x=.1,y=0.,yaw=0.,timestamp_s=0.),
              dict(robot=1,kind='landmark',index=6,cluster=0,x=3.2,y=4.)]
    events = [dict(kind='pose',robot=1,pose_index=1,from_pose_index=0,timestamp_s=1.,
                   available_time_s=1.,support_max_time_s=1.,delta=[.5,0.,0.],duration_s=1.),
              dict(kind='observation',robot=1,pose_index=1,timestamp_s=1.,available_time_s=1.,support_max_time_s=1.,
                   landmark_subject=6,source_row=10,bearing_rad=1.,range_m=4.8)]
    mean = polar_model([.6,0.,0.],[3.2,4.])[0]
    predicted = dict(checkpoint_s=0.,horizon_s=5.,predictions=[dict(robot=1,source_row=10,timestamp_s=1.,
        subject=6,mean=mean.tolist(),innovation=(np.array([1.,4.8])-mean).tolist())])
    series = RobotGroundTruthSeries(1,np.array([0.,1.,5.]),np.array([[0.,0.],[.5,0.],[2.5,0.]]),np.zeros(3))
    truth = SimpleNamespace(robots={1:series},landmarks={6:np.array([3.,4.])})
    rows = audit_window(package,states,predicted,events,[0.,0.,0.],truth,{})
    assert len(rows) == 1 and rows[0]['forecast_parity_max_abs'] < 1e-12
    assert np.linalg.norm(rows[0]['components']['motion']) < 1e-12
    assert np.linalg.norm(rows[0]['components']['checkpoint']) > .01
    predicted['predictions'][0]['mean'][1] += 1.
    with pytest.raises(ValueError,match='parity'):
        audit_window(package,states,predicted,events,[0.,0.,0.],truth,{})


def test_stratification_reports_empty_evidence_and_does_not_count_pairs_as_independent():
    from experiments.audit_predictive_error_sources import stratify
    rows = [dict(dataset=1,arm='joint_qr',robot=1,subject=6,
                 components={'measurement_reference':[0.,.1]})]
    pairs = [dict(dataset=1,arm='joint_qr',pairs=[
        dict(robot=1,source_row_a=1,source_row_b=2,normalized_squared=1.,difference_m=.1),
        dict(robot=1,source_row_a=1,source_row_b=3,normalized_squared=9.,difference_m=.3)])]
    result = stratify(rows,pairs)
    one = next(r for r in result['pairs'] if r['robot']==1)
    assert one['pair_count'] == 2 and one['unique_current_observations'] == 3
    assert one['top_ten_percent_energy_fraction'] == pytest.approx(.9)
    assert next(r for r in result['pairs'] if r['robot']==2)['pair_count'] == 0
    assert result['reference_by_subject'][0]['range']['mean'] == .1
