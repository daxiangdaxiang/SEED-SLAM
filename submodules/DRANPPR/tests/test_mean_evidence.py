import copy
import json
from pathlib import Path

import numpy as np
import pytest


def test_offset_mapping_and_robot_object_identity():
    from experiments.audit_mean_evidence import range_directions
    data = dict(schema='dranppr_physical_linearization_v1', truth_used=False,
        factors=[dict(kind='observation', robot=r, b=[0., 0.],
            terms=[dict(kind='l', index=o, a=[[1., 0.], [0., 10.]])])
            for r, o in [(1, 4), (1, 5), (2, 5)]])
    b, names = range_directions(data)
    assert names == ['robot_1', 'robot_2', 'object_4', 'object_5']
    np.testing.assert_array_equal(b.sum(axis=0), [20., 10., 10., 20.])
    np.testing.assert_array_equal(b[:, :2].sum(axis=1), b[:, 2:].sum(axis=1))
    np.testing.assert_array_equal(b[::2], 0.)
    bad = copy.deepcopy(data)
    bad['truth_used'] = True
    with pytest.raises(ValueError):
        range_directions(bad)
    bad = copy.deepcopy(data)
    bad['factors'][0]['terms'][0]['a'][1] = [0., 0.]
    with pytest.raises(ValueError):
        range_directions(bad)


def test_projection_matches_ols_and_annihilates_state():
    from experiments.audit_mean_evidence import residual_geometry
    rng = np.random.default_rng(41)
    a, b = rng.normal(size=(40, 7)), rng.normal(size=(40, 4))
    c, t, u, diagnostic = residual_geometry(a, b, 2)
    np.testing.assert_allclose(c.T @ a, 0., atol=1e-12)
    projected = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(t.T @ t, projected.T @ projected, atol=1e-12)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(u @ u.T @ t[:, :2], t[:, :2], atol=1e-12)
    assert diagnostic['state_rank'] == 7
    assert diagnostic['residual_dof'] == 33
    assert diagnostic['joint_offset_rank'] == 4


def test_joint_offsets_report_ambiguity_instead_of_inverting():
    from experiments.audit_mean_evidence import residual_geometry
    a = np.zeros((12, 1))
    b = np.zeros((12, 4))
    for k, (r, o) in enumerate([(0, 2), (0, 3), (1, 2), (1, 3)] * 3):
        b[k, r] = b[k, o] = 1.
    _, _, _, diagnostic = residual_geometry(a, b, 2)
    assert diagnostic['joint_offset_rank'] == 3
    assert diagnostic['joint_offset_nullity'] == 1


def test_f_score_scale_invariance_and_exact_scale_decomposition():
    from experiments.audit_mean_evidence import sample_scores
    rng = np.random.default_rng(42)
    z = rng.normal(size=(20, 100))
    u = np.eye(20)[:, :3]
    one, four = sample_scores(z, u), sample_scores(2*z, u)
    np.testing.assert_allclose(one['f'], four['f'], atol=1e-14)
    np.testing.assert_allclose(four['scale_all'], 4*one['scale_all'])
    np.testing.assert_allclose(one['scale_all']*20,
        one['mean_energy'] + one['scale_orthogonal']*17)
    with pytest.raises(ValueError):
        sample_scores(z, np.ones((20, 1)))


def test_predictive_score_uses_fitted_mean_and_independent_check():
    from experiments.audit_mean_evidence import predictive_scores
    train = np.array([[2., 3.], [1., 2.], [3., 4.]])
    check = np.array([[1., 5.], [2., 1.], [4., 2.]])
    u = np.eye(3)[:, :1]
    out = predictive_scores(train, check, u)
    q0 = np.mean(train**2, axis=0)
    mean = u @ u.T @ train
    q1 = np.sum((train-mean)**2, axis=0)/2
    np.testing.assert_allclose(out['scalar_nll'],
        .5*(np.log(2*np.pi*q0)+np.mean(check**2, axis=0)/q0))
    np.testing.assert_allclose(out['mean_nll'],
        .5*(np.log(2*np.pi*q1)+np.mean((check-mean)**2, axis=0)/q1))


def test_paired_simulation_is_repeatable_and_matches_exact_noise_null():
    from experiments.audit_mean_evidence import simulate
    t = np.zeros((20, 4))
    t[0, 0] = t[1, 1] = t[0, 2] = t[1, 3] = 10.
    names = ['robot_1', 'robot_2', 'object_1', 'object_2']
    u = np.eye(20)[:, :2]
    one = simulate(t, names, u, 4096, 20261027)
    assert one == simulate(t, names, u, 4096, 20261027)
    rows = {r['case']: r for r in one}
    null = rows['gaussian_1']
    assert abs(null['f_reject_fraction']-.05) < .015
    assert rows['gaussian_4']['f_reject_fraction'] == null['f_reject_fraction']
    assert rows['state_equivalent'] == dict(null, case='state_equivalent')
    np.testing.assert_allclose(rows['robot_1_0.30m']['expected_scale_all'], 1+9/20)


def test_rejects_nonfinite_or_invalid_bias_dimensions():
    from experiments.audit_mean_evidence import residual_geometry
    with pytest.raises(ValueError):
        residual_geometry(np.ones((10, 2)), np.zeros((9, 3)), 2)
    with pytest.raises(ValueError):
        residual_geometry(np.full((10, 2), np.nan), np.zeros((10, 3)), 2)


def test_fully_state_equivalent_robot_offsets_are_not_testable():
    from experiments.audit_mean_evidence import residual_geometry
    rng = np.random.default_rng(43)
    a = rng.normal(size=(30, 5))
    directions = a @ rng.normal(size=(5, 4))
    with pytest.raises(ValueError, match='testable'):
        residual_geometry(a, directions, 2)


def test_archived_native_range_units_and_single_observation_null_directions():
    from experiments.audit_mean_evidence import range_directions, residual_geometry
    from experiments.network_noise_support import assemble
    root = Path(__file__).resolve().parents[1]
    source = root/'results/dranppr_network_noise_transfer_qr_20261003/d4/training/linearization.json'
    data = json.loads(source.read_text())
    a, _, _ = assemble(data)
    b, names = range_directions(data)
    np.testing.assert_allclose(b[b != 0.], 10., rtol=1e-12)
    c, t, _, diagnostic = residual_geometry(a, b, 5)
    assert diagnostic['joint_offset_nullity'] == 4
    assert diagnostic['robot_offset_rank'] == 5
    for name in ('object_2', 'object_4', 'object_15'):
        column = names.index(name)
        assert np.count_nonzero(b[:, column]) == 1
        assert np.linalg.norm(t[:, column]) < 1e-11
    perturbation = np.sin(np.arange(a.shape[1]))*.01
    residual = a @ perturbation + np.arange(len(a))*.001
    np.testing.assert_allclose(c.T @ residual,
        c.T @ (np.arange(len(a))*.001), atol=1e-12)
