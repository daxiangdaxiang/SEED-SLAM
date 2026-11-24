import math

import numpy as np
import pytest


def test_information_covariance_matches_known_independent_variance_groups():
    from experiments.audit_noise_parameter_uncertainty import parameter_covariance
    components = np.array([np.diag([1.,1.,0.]), np.diag([0.,0.,1.])])
    covariance, information = parameter_covariance(np.zeros((3,3)), components, np.array([2.,7.]))
    assert np.allclose(information, np.diag([1.,.5]))
    assert np.allclose(covariance, np.diag([1.,2.]))


def test_cubature_preserves_log_parameter_mean_and_covariance():
    from experiments.audit_noise_parameter_uncertainty import cubature
    mean = np.array([1.,2.])
    covariance = np.array([[2.,.3],[.3,1.]])
    points = cubature(mean,covariance)
    assert points.shape == (4,2)
    assert np.allclose(points.mean(axis=0),mean)
    delta = points-mean
    assert np.allclose(delta.T@delta/len(points),covariance)


def row(mean_range, measurement=2.):
    energy = (measurement-mean_range)**2
    return dict(robot=1,source_row=2,timestamp_s=3.,subject=6,
        mean=[0.,mean_range],innovation=[0.,measurement-mean_range],
        covariance=np.eye(2).tolist(),nll=.5*energy+math.log(2*math.pi),energy=energy)


def test_mixture_uses_density_average_not_average_nll_and_keeps_total_variance():
    from experiments.audit_noise_parameter_uncertainty import mix_predictions
    original = row(1.)
    assert mix_predictions([[original]])[0]['nll'] == pytest.approx(original['nll'])
    result = mix_predictions([[row(1.)],[row(3.)]])[0]
    assert result['mean'][1] == 2.
    assert result['covariance'][1][1] == pytest.approx(2.)
    assert result['nll'] == pytest.approx(.5+math.log(2*math.pi))
    expected = -math.log((math.exp(-row(1.)['nll'])+math.exp(-row(2.)['nll']))/2)
    assert mix_predictions([[row(1.)],[row(2.)]])[0]['nll'] == pytest.approx(expected)
    with pytest.raises(ValueError):
        mix_predictions([[row(1.)],[dict(row(2.),source_row=3)]])
