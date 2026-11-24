import pytest

from experiments.audit_native_chart_refresh import summarize_records


def test_summary_preserves_stage_robot_and_whitened_error_denominators():
    family=dict(count=2,residual_dimensions=4,residual_change_sq=16.,raw_error_sq=4.,
        transported_error_sq=1.,target_residual_sq=25.,raw_error_max=1.5,
        transported_error_max=.7,gauge_invariance_max=1e-13,cached_residual_max=0.,
        raw_error_bins=[0,2],transported_error_bins=[1,1],residual_change_bins=[0,2])
    record=dict(robot=2,stage='optimizer_refresh',epoch=0,pose_count=3,point_count=1,
        requested_pose_keys=2,requested_point_keys=1,raw_trigger_pose_keys=2,raw_trigger_point_keys=1,
        aligned_trigger_pose_keys=2,aligned_trigger_point_keys=1,pose_translation_raw_sq=10.,
        pose_translation_aligned_sq=2.,max_raw_delta=2.,max_aligned_delta=1.,fixed_aligned_max=0.,
        factors=2,touched_factors=2,pose_only_factors=1,point_only_factors=0,both_factors=1,
        norm_bin_upper_bounds=[.1],families={'observation':family})
    result=summarize_records([record,dict(record,stage='identity_chart',robot=3)])
    assert result['records']==2
    assert result['stages']['optimizer_refresh']['touched_factors']==2
    assert result['stages']['identity_chart']['touched_factors']==2
    group=result['robots']['2']['optimizer_refresh']
    assert group['pose_translation_removed_fraction']==pytest.approx(.8)
    assert group['families']['observation']['raw_error_rms_per_dimension']==pytest.approx(1.)
    assert group['families']['observation']['transported_error_rms_per_dimension']==pytest.approx(.5)
    assert result['stages']['optimizer_refresh']['families']['observation']['raw_error_bins']==[0,2]
    record['pose_origin']=dict(record,pose_translation_aligned_sq=5.)
    origin=summarize_records([record])['pose_origin']
    assert origin['stages']['optimizer_refresh']['pose_translation_removed_fraction']==pytest.approx(.5)
    with pytest.raises(ValueError,match='bins'):
        summarize_records([record,dict(record,norm_bin_upper_bounds=[1.])])
