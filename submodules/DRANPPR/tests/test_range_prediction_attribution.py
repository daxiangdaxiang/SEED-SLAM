import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest


def prediction(mean=(.1,2.), covariance=((.04,.01),(.01,.09)), row=1, timestamp=31.):
    from experiments.analyze_motion_time_structure import split_score
    innovation=[math.atan2(math.sin(.3-mean[0]),math.cos(.3-mean[0])),2.4-mean[1]]
    return dict(robot=1,source_row=row,subject=7,timestamp_s=timestamp,horizon_s=timestamp-30.,
        mean=list(mean),innovation=innovation,covariance=[list(r) for r in covariance],
        observation_covariance=[[.01,0.],[0.,.02]],nll=split_score(innovation,covariance)['joint'])


def test_pair_swaps_match_scalar_scores_and_preserve_inputs():
    from experiments.audit_range_prediction_attribution import paired_prediction
    a=prediction();b=prediction((.2,2.2),((.03,.008),(.008,.05)))
    original=copy.deepcopy((a,b));r=paired_prediction(a,b)
    assert len(r['scores'])==8 and (a,b)==original
    for m in (0,1):
        for u in (0,1):
            for obs in (0,1):
                forecasts=(a,b)
                error=forecasts[m]['innovation'][1]
                variance=(forecasts[u]['covariance'][1][1]-forecasts[u]['observation_covariance'][1][1]
                          +forecasts[obs]['observation_covariance'][1][1])
                s=r['scores'][f'm{m}_u{u}_r{obs}']
                expected=.5*(error**2/variance+np.log(2*np.pi*variance))
                assert s['range']==pytest.approx(expected,abs=1e-14)
                assert s['joint']==pytest.approx(s['bearing']+s['range_given_bearing'],abs=1e-14)
    assert r['scores']['m0_u0_r0']['joint']==a['nll']
    assert r['scores']['m1_u1_r1']['joint']==b['nll']


def test_pair_reconstructs_bearing_modulo_wrap_and_checks_observed_values():
    from experiments.audit_range_prediction_attribution import paired_prediction
    a=prediction();b=prediction((.2+2*np.pi,2.2))
    assert paired_prediction(a,b)['max_observation_difference']<1e-12
    b['mean'][1]+=.01
    with pytest.raises(ValueError,match='observation'):paired_prediction(a,b)


def test_pair_rejects_invalid_covariance_decomposition_and_score():
    from experiments.audit_range_prediction_attribution import paired_prediction
    a=prediction();b=prediction()
    b['observation_covariance'][1][1]=.2
    with pytest.raises(ValueError,match='covariance'):paired_prediction(a,b)
    b=prediction();b['nll']+=.01
    with pytest.raises(ValueError,match='score'):paired_prediction(a,b)
    b=prediction();b['covariance'][0][1]=.02
    with pytest.raises(ValueError,match='covariance'):paired_prediction(a,b)


def test_support_matching_rejects_missing_duplicate_or_truth_assisted_rows():
    from experiments.audit_range_prediction_attribution import matched_predictions
    a=dict(truth_used=False,predictions=[prediction(),prediction(row=2)])
    b=dict(truth_used=False,predictions=list(reversed(a['predictions'])))
    assert len(matched_predictions(a,b))==2
    with pytest.raises(ValueError,match='support'):
        matched_predictions(a,dict(b,predictions=b['predictions'][:1]))
    with pytest.raises(ValueError,match='duplicate'):
        matched_predictions(a,dict(b,predictions=[prediction(),prediction()]))
    with pytest.raises(ValueError,match='truth'):
        matched_predictions(a,dict(b,truth_used=True))


def test_group_frame_weighting_and_harm_mass_are_not_independent_samples():
    from experiments.audit_range_prediction_attribution import paired_prediction,summarize_rows
    rows=[]
    for i,time,mean in ((1,31.,2.1),(2,31.,2.1),(3,32.,2.6)):
        a=prediction(row=i,timestamp=time);b=prediction((.1,mean),row=i,timestamp=time)
        rows.append(dict(dataset=1,elapsed=30,**paired_prediction(a,b)))
    summary=summarize_rows(rows)
    assert summary['observations']==3 and summary['frames']==2
    key='m1_u1_r1'
    expected=(rows[0]['scores'][key]['range']+rows[2]['scores'][key]['range'])/2
    assert summary['frame_equal_scores'][key]['range']==pytest.approx(expected)
    assert summary['frame_equal_scores'][key]['range']!=pytest.approx(summary['scores'][key]['range'])
    harm=summary['range_harm']
    delta=summary['scores'][key]['range']-summary['scores']['m0_u0_r0']['range']
    assert harm['positive_mass']-harm['negative_mass']==pytest.approx(3*delta)


def test_group_reports_mean_covariance_interaction_exactly():
    from experiments.audit_range_prediction_attribution import paired_prediction,summarize_rows
    a=prediction();b=prediction((.2,2.2),((.03,.008),(.008,.05)))
    r=summarize_rows([dict(dataset=1,elapsed=30,**paired_prediction(a,b))])
    effect=r['range_effects']
    assert effect['total']==pytest.approx(effect['mean_at_old_covariance']+
        effect['covariance_at_old_mean']+effect['interaction'])
    assert effect['total']==pytest.approx(effect['mean_at_new_covariance']+
        effect['covariance_at_old_mean'])
    assert effect['total']==pytest.approx(effect['covariance_at_new_mean']+
        effect['mean_at_old_covariance'])
    assert abs(effect['interaction'])>1e-3


def test_source_event_audit_accounts_for_unknown_objects_and_rejects_future_state():
    from experiments.audit_range_prediction_attribution import audit_forecast_support
    p=prediction()
    forecast=dict(truth_used=False,checkpoint_s=30.,horizon_s=5.,predictions=[p],unknown_object_observations=1)
    package=dict(truth_used=False,robots=[dict(id=1,poses=[dict(timestamp_s=30.)],objects=[dict(label=7)])])
    event=dict(kind='observation',robot=1,source_row=1,landmark_subject=7,timestamp_s=31.,
        available_time_s=31.,support_max_time_s=31.,bearing_rad=.3,range_m=2.4)
    events=[event,dict(event,source_row=2,landmark_subject=8)]
    result=audit_forecast_support(forecast,package,events)
    assert result['known']==1 and result['unknown']==1 and result['total']==2
    assert result['max_observation_difference']<1e-12
    bad=copy.deepcopy(events);bad[0]['range_m']+=.01
    with pytest.raises(ValueError,match='observation'):audit_forecast_support(forecast,package,bad)
    package['robots'][0]['poses'][0]['timestamp_s']=34.
    with pytest.raises(ValueError,match='future'):audit_forecast_support(forecast,package,events)


def test_real_range_audit_matches_all_archived_support(tmp_path):
    from experiments.audit_range_prediction_attribution import run
    root=Path(__file__).resolve().parents[1]
    source=root/'results/dranppr_network_noise_transfer_qr_20261003'
    previous=root/'results/dranppr_motion_time_structure_verified_20261004'
    if not all((p/'SHA256SUMS').is_file() for p in (source,previous)):
        pytest.skip('frozen diagnostic archives unavailable')
    output=tmp_path/'range'
    run(source,previous,output)
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['truth_used'] and not manifest['parameters_fitted']
    rows=json.loads((output/'rows.json').read_text())['rows']
    groups=json.loads((output/'summary.json').read_text())
    assert len(rows)==533 and len(groups['robot'])==10 and len(groups['frame'])==386
    assert sum(len(r['scores']) for r in rows)==4264
    assert manifest['max_score_parity']<1e-10 and manifest['max_previous_group_parity']<1e-10
    assert len(groups['solver_checks'])==12
    assert len(groups['support_checks'])==12
    assert sum(r['unknown'] for r in groups['support_checks'] if r['arm']=='nominal')==17
