import copy
import json

import numpy as np
import pytest

from test_native_joint_audit import package
from test_network_noise_support import exported


def test_deletion_groups_preserve_whole_frames_and_subjects(tmp_path):
    from experiments.audit_noise_block_sensitivity import deletion_groups, retained_model
    out = exported(tmp_path)
    data = json.loads((out/'linearization.json').read_text())
    original = copy.deepcopy(data)
    plan = deletion_groups(data, package())
    assert [v['tag'] for v in plan] == ['full', 'subject_6', 'subject_7', 'subject_8',
                                       'frames_0', 'frames_1', 'frames_2']
    observations = {i for i,f in enumerate(data['factors']) if f['kind']=='observation'}
    for family in ['subject','frames']:
        blocks = [set(v['removed']) for v in plan if v['family']==family]
        assert set.union(*blocks)==observations
        assert sum(map(len,blocks))==len(observations)
    for group in plan[1:]:
        reduced = retained_model(data, group['removed'])
        assert sum(f['kind']=='odometry' for f in reduced['factors'])==4
        assert len(reduced['factors'])+len(group['removed'])==len(data['factors'])
        assert group['removed_observations']==6
        if group['family']=='frames':
            assert group['removed_frames']==2
        else:
            assert group['removed_frames']==6
    assert data==original


def test_subject_grouping_uses_declared_labels_across_clusters(tmp_path):
    from experiments.audit_noise_block_sensitivity import deletion_groups
    out=exported(tmp_path)
    data=json.loads((out/'linearization.json').read_text())
    p=package()
    for f in data['factors']:
        if f['robot']==2:
            for t in f['terms']:
                if t['kind']=='l':t['index']+=10
    for obj in p['robots'][1]['objects']:obj['cluster']+=10
    for obs in p['robots'][1]['observations']:obs['cluster']+=10
    plan=deletion_groups(data,p)
    assert [g['removed_observations'] for g in plan if g['family']=='subject']==[6,6,6]


def test_deletion_rejects_unmatched_metadata_and_odometry(tmp_path):
    from experiments.audit_noise_block_sensitivity import deletion_groups, retained_model
    out=exported(tmp_path)
    data=json.loads((out/'linearization.json').read_text())
    p=package()
    p['robots'][0]['observations'].pop()
    with pytest.raises(ValueError,match='match'):
        deletion_groups(data,p)
    index=next(i for i,f in enumerate(data['factors']) if f['kind']=='odometry')
    with pytest.raises(ValueError,match='observation'):
        retained_model(data,[index])
    with pytest.raises(ValueError):
        retained_model(data,[0,0])
    data['truth_used']=True
    with pytest.raises(ValueError,match='truth'):
        deletion_groups(data,package())


def test_fixed_linear_refit_is_invariant_to_state_column_shift(tmp_path):
    from experiments.audit_noise_block_sensitivity import fit_model
    from experiments.network_noise_support import assemble
    out=exported(tmp_path)
    data=json.loads((out/'linearization.json').read_text())
    rng=np.random.default_rng(8104)
    for f in data['factors']:f['b']=rng.normal(size=len(f['b'])).tolist()
    a,b,_=assemble(data)
    shifted=copy.deepcopy(data)
    value=b+a@rng.normal(size=a.shape[1])
    offset=0
    for f in shifted['factors']:
        count=len(f['b']); f['b']=value[offset:offset+count].tolist();offset+=count
    one,two=fit_model(data),fit_model(shifted)
    assert one['success'] and two['success']
    assert one['component_rank']==5 and one['nullity']==0
    assert one['residual_dof']==27
    assert np.allclose(one['scales'],two['scales'],rtol=2e-5,atol=1e-6)
    assert one['nll']==pytest.approx(two['nll'],abs=1e-8)


def test_forecast_comparison_requires_exact_shared_support():
    from experiments.audit_noise_block_sensitivity import compare_predictions
    one=[dict(robot=1,source_row=9,subject=7,timestamp_s=31.,nll=-2.,energy=1.),
         dict(robot=2,source_row=10,subject=8,timestamp_s=32.,nll=0.,energy=8.)]
    two=[dict(one[0],nll=-1.,energy=2.),dict(one[1],nll=-1.,energy=3.)]
    result=compare_predictions(one,two)
    assert result['delta_mean_nll']==0.
    assert result['candidate']['coverage_95']==.5
    assert result['by_robot'][0]['delta_mean_nll']==-1.
    with pytest.raises(ValueError,match='support'):
        compare_predictions(one,[dict(two[0],source_row=19),two[1]])
    with pytest.raises(ValueError,match='support'):
        compare_predictions(one,one+one)
