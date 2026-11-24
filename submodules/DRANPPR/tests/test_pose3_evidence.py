import numpy as np
import pytest


def test_independent_repetitions_add_noise_evidence_not_state_columns_or_test_observations():
    from experiments.audit_pose3_evidence import sample_control
    a = np.arange(24.).reshape(8,3)
    labels,q = np.arange(8)%2,np.array([.1,4.])
    one = sample_control(a,labels,q,401,1)
    four = sample_control(a,labels,q,401,4)
    np.testing.assert_array_equal(four['a'].toarray(),np.tile(a,(4,1)))
    np.testing.assert_array_equal(four['labels'],np.tile(labels,4))
    np.testing.assert_array_equal(one['training_y'],four['training_y'][:8])
    np.testing.assert_array_equal(one['test_y'],four['test_y'])
    assert not np.array_equal(four['training_y'][:8],four['training_y'][8:16])
    assert not np.array_equal(one['training_y'],one['test_y'])
    np.testing.assert_array_equal(four['training_y'],sample_control(a,labels,q,401,4)['training_y'])


@pytest.mark.parametrize('count',[0,-1,1.5])
def test_control_rejects_invalid_repetitions(count):
    from experiments.audit_pose3_evidence import sample_control
    with pytest.raises(ValueError):
        sample_control(np.ones((8,2)),np.zeros(8,dtype=int),np.ones(1),401,count)


def test_prediction_risk_is_state_error_in_noise_units_and_column_scale_invariant():
    from experiments.audit_pose3_evidence import prediction_risk
    a = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-1.]])
    state = np.array([.2,-.5])
    y = a@state
    variance = np.array([1.,2.,3.,4.])
    expected = np.mean(y*y/variance)
    assert prediction_risk(a,y,variance,variance) == pytest.approx(expected,rel=1e-12)
    assert prediction_risk(a*np.array([1e4,1e-3]),y,variance,variance) == pytest.approx(expected,rel=1e-10)
    assert prediction_risk(a,np.zeros(4),variance,variance) == pytest.approx(0.,abs=1e-15)


def test_independent_dense_qr_check_matches_error_contrasts():
    from experiments.audit_pose3_evidence import dense_qr_score
    from dranppr_variance_components import ContrastVarianceModel
    rng = np.random.default_rng(73)
    a,y = rng.normal(size=(24,5)),rng.normal(size=24)
    d = np.eye(2)[np.arange(24)%2]
    q = np.array([.1,4.])
    reference = ContrastVarianceModel(a,y,d,np.zeros(24))
    value,score = dense_qr_score(a,y,d,q)
    assert value == pytest.approx(reference.nll(np.log(q)),rel=0.,abs=1e-10)
    np.testing.assert_allclose(score,reference.gradient(np.log(q)),rtol=0.,atol=1e-10)
