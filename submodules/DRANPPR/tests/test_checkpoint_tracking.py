import pytest
import numpy as np


def test_latest_pose_selection_uses_time_not_row_order_and_ignores_objects():
    from experiments.audit_checkpoint_tracking import latest_pose_rows
    rows = [dict(robot='1',kind='pose',index='4',timestamp_s='9'),
            dict(robot='2',kind='pose',index='3',timestamp_s='8'),
            dict(robot='1',kind='pose',index='2',timestamp_s='7'),
            dict(robot='1',kind='landmark',index='8',timestamp_s='99')]
    assert latest_pose_rows(rows,[1,2]) == rows[:2]


def test_latest_pose_selection_rejects_duplicate_and_missing_support():
    from experiments.audit_checkpoint_tracking import latest_pose_rows
    row = dict(robot='1',kind='pose',index='4',timestamp_s='9')
    with pytest.raises(ValueError):
        latest_pose_rows([row],[1,2])
    with pytest.raises(ValueError):
        latest_pose_rows([row,row],[1])


def checkpoint_inputs():
    manifest = {'benchmark_cases':[{'dataset':9,'duration':300}]}
    warm = {'provenance':{'dataset':9}, 'robots':[
        {'poses':[{'timestamp_s':1000.},{'timestamp_s':1131.723}]},
        {'poses':[{'timestamp_s':1001.},{'timestamp_s':1130.}]}]}
    return manifest,warm


def test_checkpoints_follow_declared_dataset_prefix_and_horizon():
    from experiments.audit_checkpoint_tracking import checkpoint_times
    manifest,warm = checkpoint_inputs()
    assert checkpoint_times(manifest,warm,9,[150,210,270]) == [150.,210.,270.]
    manifest = {'benchmark_cases':[{'dataset':1,'duration':120}]}
    warm['provenance']['dataset'] = 1
    for robot in warm['robots']:
        robot['poses'][-1]['timestamp_s'] = 1015.
    assert checkpoint_times(manifest,warm,1,None) == [30.,60.,90.]


@pytest.mark.parametrize('times', [[],[130],[150,150],[210,150],[301],
                                   [float('nan')],[float('inf')]])
def test_invalid_checkpoints_are_rejected(times):
    from experiments.audit_checkpoint_tracking import checkpoint_times
    manifest,warm = checkpoint_inputs()
    with pytest.raises(ValueError):
        checkpoint_times(manifest,warm,9,times)


def test_checkpoint_dataset_must_match_declared_source_and_prefix():
    from experiments.audit_checkpoint_tracking import checkpoint_times
    manifest,warm = checkpoint_inputs()
    with pytest.raises(ValueError):
        checkpoint_times(manifest,warm,1,[150])
    warm['provenance']['dataset'] = 1
    with pytest.raises(ValueError):
        checkpoint_times(manifest,warm,9,[150])


def test_completion_geometry_separates_rigid_movement_from_deformation():
    from experiments.audit_checkpoint_tracking import completion_geometry
    points = np.array([[0.,0.],[2.,0.],[0.,1.],[2.,1.]])
    def rows(x):
        return [dict(robot=1,kind='pose',index=i,x=p[0],y=p[1]) for i,p in enumerate(x)]
    angle = .3
    r = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rigid = completion_geometry(rows(points),rows(points@r.T+np.array([1.,2.])))
    assert rigid['deformation_rms_m'] < 1e-12
    assert rigid['rigid_explained_fraction'] == pytest.approx(1.)
    assert rigid['rotation_rad'] == pytest.approx(angle)
    changed = completion_geometry(rows(points),rows(points*np.array([1.4,1.])))
    assert changed['deformation_rms_m'] > .1
    assert changed['rigid_explained_fraction'] < .9


def test_completion_geometry_rejects_different_or_duplicated_pose_support():
    from experiments.audit_checkpoint_tracking import completion_geometry
    row = dict(robot=1,kind='pose',index=0,x=0.,y=0.)
    other = dict(row,index=1,x=1.)
    with pytest.raises(ValueError):
        completion_geometry([row,other],[row])
    with pytest.raises(ValueError):
        completion_geometry([row,other],[row,other,other])
