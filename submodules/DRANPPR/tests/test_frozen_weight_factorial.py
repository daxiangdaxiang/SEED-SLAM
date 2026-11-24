import copy

import numpy as np
import pytest

from test_range_prediction_attribution import prediction


def test_frozen_grid_changes_only_declared_weight_groups():
    from experiments.audit_frozen_weight_factorial import weight_grid
    scales=[.2,.4,.8,.003,.7];original=scales[:]
    grid=weight_grid(scales)
    assert len(grid)==8 and scales==original
    assert grid['q0_b0_r0']==[1.]*5 and grid['q1_b1_r1']==scales
    for q in (0,1):
        for b in (0,1):
            for r in (0,1):
                assert grid[f'q{q}_b{b}_r{r}']==(scales[:3] if q else [1.]*3)+[
                    scales[3] if b else 1.,scales[4] if r else 1.]
    with pytest.raises(ValueError):weight_grid([1,1,1,0,1])


def test_endpoint_parity_checks_support_means_and_covariances():
    from experiments.audit_frozen_weight_factorial import endpoint_parity
    a=dict(truth_used=False,predictions=[prediction()])
    assert endpoint_parity(a,copy.deepcopy(a))==dict(max_mean=0.,max_covariance=0.,max_nll=0.)
    b=copy.deepcopy(a);b['predictions'][0]['mean'][0]+=2*np.pi
    assert endpoint_parity(a,b)['max_mean']<1e-12
    b=copy.deepcopy(a);b['predictions'][0]['mean'][1]+=.01
    with pytest.raises(ValueError,match='parity'):endpoint_parity(a,b)
    b=copy.deepcopy(a);b['predictions'][0]['source_row']+=1
    with pytest.raises(ValueError,match='support'):endpoint_parity(a,b)
