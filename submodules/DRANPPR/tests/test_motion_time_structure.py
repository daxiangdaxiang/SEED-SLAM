import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_canonical_controls_match_existing_duplicate_averaging():
    from experiments.audit_motion_time_structure import canonical_control_table
    table=np.array([[0.,1.,0.],[1.,5.,.2],[1.,2.,.4],[2.,9.,0.]])
    original=table.copy()
    merged=canonical_control_table(table)
    assert np.allclose(merged,[[0.,1.,0.],[1.,3.5,.3],[2.,9.,0.]])
    assert np.array_equal(table,original)


def test_raw_command_integration_is_causal_and_uses_last_duplicate():
    from experiments.audit_motion_time_structure import HeldCommands
    table=np.array([[0.,1.,0.],[1.,5.,0.],[1.,2.,0.],[2.,9.,0.],[3.,0.,0.]])
    one=HeldCommands(table).increment(.5,1.5)
    assert one['delta']==pytest.approx([1.5,0.,0.])
    changed=table.copy();changed[3:,1:]=100.
    assert HeldCommands(changed).increment(.5,1.5)==one
    assert one['last_control_time_s']==1.
    with pytest.raises(ValueError):HeldCommands(table).increment(-1.,1.)
    with pytest.raises(ValueError):HeldCommands(table[::-1])


def test_command_composition_preserves_unwrapped_rotation():
    from experiments.audit_motion_time_structure import HeldCommands
    from benchmarks.mrclam.extract import PlanarPose
    controls=HeldCommands(np.array([[0.,1.,2.],[4.,1.,2.]]))
    total=controls.increment(0.,4.)
    first=controls.increment(0.,2.);second=controls.increment(2.,4.)
    pose=PlanarPose(*first['delta']).compose(PlanarPose(*second['delta']))
    assert total['delta']==pytest.approx([pose.x,pose.y,pose.yaw],abs=1e-14)
    assert total['yaw_integral_rad']==8.
    assert abs(total['delta'][2])<math.pi


def test_whitened_energy_decomposes_rate_and_scatter_for_unequal_durations():
    from experiments.audit_motion_time_structure import increment_statistics
    dt=np.array([.2,.5,1.3])
    error=np.array([.01,-.02,.08]);density=.04
    result=increment_statistics(error,dt,density)
    assert result['rate_radps']==pytest.approx(error.sum()/dt.sum())
    assert result['total_energy']==pytest.approx(np.sum(error**2/dt)/density)
    assert result['total_energy']==pytest.approx(result['rate_energy']+result['scatter_energy'])
    drift=increment_statistics(.03*dt,dt,density)
    assert drift['scatter_energy']<1e-20 and drift['rate_energy_fraction']==pytest.approx(1.)
    with pytest.raises(ValueError):increment_statistics(error,[0.,1.,1.],density)


def test_disjoint_blocks_expose_linear_and_quadratic_growth_without_overlap():
    from experiments.audit_motion_time_structure import horizon_statistics
    result=horizon_statistics(np.full(12,.02),.0025)
    assert [r['seconds'] for r in result]==[1,2,5,10]
    for r in result:
        assert r['blocks']==12//r['seconds']
        assert r['unused_seconds']==12%r['seconds']
        assert r['rmse_rad']==pytest.approx(.02*r['seconds'])
        assert r['mean_energy']==pytest.approx(.02**2*r['seconds']/.0025)


def test_lag_statistic_is_on_increments_and_handles_degenerate_series():
    from experiments.audit_motion_time_structure import lag_correlation
    assert lag_correlation([1.,-1.,1.,-1.,1.,-1.],1)==pytest.approx(-1.)
    assert lag_correlation([1.,1.,1.,1.],1) is None
    assert lag_correlation([1.,2.],1) is None


def test_motion_window_uses_actual_state_age_not_observation_count():
    from benchmarks.mrclam.evaluate import RobotGroundTruthSeries
    from experiments.audit_motion_time_structure import window_motion
    package=dict(robots=[dict(id=1,poses=[dict(index=0,timestamp_s=0.)])])
    states=[dict(kind='pose',robot=1,index=0,x=0.,y=0.,yaw=.1)]
    truth=SimpleNamespace(robots={1:RobotGroundTruthSeries(1,np.array([0.,4.]),
        np.array([[0.,0.],[4.,0.]]),np.zeros(2))})
    events=[dict(kind='pose',robot=1,pose_index=1,from_pose_index=0,timestamp_s=3.,
        available_time_s=3.,support_max_time_s=3.,delta=[3.,0.,.2],duration_s=3.),
        dict(kind='observation',robot=1,pose_index=1,timestamp_s=3.),
        dict(kind='observation',robot=1,pose_index=1,timestamp_s=3.)]
    rows=window_motion(package,states,events,truth,[0.,0.,0.],2.,1.,[1.,1.,2.])
    assert len(rows)==1
    r=rows[0]
    assert r['state_age_s']==2. and r['propagation_s']==3. and r['forecast_horizon_s']==1.
    assert r['motion_yaw_error_rad']==pytest.approx(.2)
    assert r['initial_yaw_error_rad']==pytest.approx(.1)
    assert r['total_yaw_error_rad']==pytest.approx(.3)
    assert r['process_yaw_variance']==pytest.approx(.05**2*2.*3.)
    events[0]['duration_s']=1.
    with pytest.raises(ValueError,match='duration'):
        window_motion(package,states,events,truth,[0.,0.,0.],2.,1.,[1.,1.,2.])


def test_frozen_motion_audit_completes_both_stages(tmp_path):
    from experiments.audit_motion_time_structure import run
    from experiments.analyze_motion_time_structure import analyze
    root=Path(__file__).resolve().parents[1]
    source=root/'results/dranppr_network_noise_transfer_qr_20261003'
    attribution=root/'results/dranppr_predictive_error_sources_20261004'
    raw=Path('/home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw')
    if not all((p/'SHA256SUMS').is_file() for p in (source,attribution)) or not raw.is_dir():
        pytest.skip('frozen MRCLAM diagnostic fixture not available')
    output=tmp_path/'audit'
    run(source,attribution,raw,output)
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['gt_free_stage_finished'] and manifest['reference_stage_finished']
    assert not manifest['parameters_fitted'] and not manifest['estimator_changed']
    rows=json.loads((output/'bearing_attribution.json').read_text())['rows']
    assert len(rows)==533
    assert len(json.loads((output/'uniform_increments.json').read_text())['rows'])==980
    summary=json.loads((output/'summary.json').read_text())
    assert summary['max_bearing_parity']<1e-10
    assert len(summary['windows'])==30 and len(summary['uniform'])==10
    analyze(output)
    scores=json.loads((output/'score_components.json').read_text())
    assert len(scores['rows'])==1066 and len(scores['groups'])==20
    assert not scores['truth_used'] and scores['max_score_parity']<1e-10


def test_gaussian_score_split_matches_both_conditional_orders():
    from experiments.analyze_motion_time_structure import split_score
    v=np.array([.2,-.4]);c=np.array([[.2,.1],[.1,.5]])
    r=split_score(v,c)
    joint=.5*(v@np.linalg.solve(c,v)+np.linalg.slogdet(c)[1]+2*np.log(2*np.pi))
    assert r['joint']==pytest.approx(joint,abs=1e-14)
    assert r['bearing']+r['range_given_bearing']==pytest.approx(joint,abs=1e-14)
    assert r['range']+r['bearing_given_range']==pytest.approx(joint,abs=1e-14)
    swapped=split_score(v[::-1],c[::-1,::-1])
    assert swapped['bearing']==pytest.approx(r['range'])
    assert swapped['range_given_bearing']==pytest.approx(r['bearing_given_range'])
    with pytest.raises(ValueError):split_score(v,[[1.,2.],[2.,1.]])


def test_gaussian_score_split_does_not_drop_covariance_coupling():
    from experiments.analyze_motion_time_structure import split_score
    r=split_score([.5,-.8],[[1.,.8],[.8,1.]])
    assert abs(r['bearing']+r['range']-r['joint'])>.1
    independent=split_score([.5,-.8],[[1.,0.],[0.,1.]])
    assert independent['bearing']+independent['range']==pytest.approx(independent['joint'])
