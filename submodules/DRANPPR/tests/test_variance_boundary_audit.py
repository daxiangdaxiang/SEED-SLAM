import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'experiments/audit_variance_boundary.py'
    assert path.is_file(), 'variance boundary diagnostic not implemented'
    spec = importlib.util.spec_from_file_location('variance_boundary', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_additive_variance_likelihood_and_jeffreys_are_not_scale_prior():
    m = module()
    logs = np.array([-32., -16., 0.])
    nll, jeffreys = m.slice_terms(np.zeros(1), np.eye(1), np.eye(1), logs)
    q = np.exp(logs)
    np.testing.assert_allclose(nll, .5*np.log(2*np.pi*(1+q)), rtol=1e-12)
    # Density w.r.t. dq: sqrt(I(q)) = 1 / (sqrt(2)*(1+q)).
    np.testing.assert_allclose(jeffreys, -.5*np.log(2)-np.log1p(q), atol=1e-12)
    assert abs(nll[0]-nll[1]) < 1e-6
    assert abs(jeffreys[0]) < 1.


def test_correct_coordinate_jacobian_matches_analytic_conditional_mean():
    m = module()
    logs = np.linspace(-32., 0., 4097)
    nll, jeffreys = m.slice_terms(np.zeros(1), np.eye(1), np.eye(1), logs)
    summary = m.truncated_summary(logs, -nll+jeffreys+logs)
    assert summary['mean_component'] == pytest.approx(np.sqrt(2)-1, abs=2e-6)


def test_log_uniform_truncation_moves_mass_toward_zero():
    m = module()
    summaries = []
    for bound in [-16., -32., -64.]:
        logs = np.linspace(bound, 0., int(-bound*64)+1)
        nll, _ = m.slice_terms(np.zeros(1), np.eye(1), np.eye(1), logs)
        summaries.append(m.truncated_summary(logs, -nll))
    assert summaries[2]['log_integral'] > summaries[1]['log_integral'] > summaries[0]['log_integral']
    assert summaries[2]['median_component'] < summaries[1]['median_component'] < summaries[0]['median_component']


def test_boundary_audit_rejects_uninformative_or_invalid_slice():
    m = module()
    with pytest.raises(ValueError, match='uninformative'):
        m.slice_terms(np.zeros(1), np.eye(1), np.zeros((1,1)), np.array([-1.,0.]))
    with pytest.raises(ValueError, match='invalid'):
        m.slice_terms(np.zeros(1), np.eye(2), np.eye(2), np.array([-1.,0.]))
