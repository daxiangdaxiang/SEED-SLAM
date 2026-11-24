import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_native_joint_audit import package
from test_frozen_noise_completion import weighted


def exported(tmp_path, flag=True):
    source = tmp_path/'input.json'
    source.write_text(json.dumps(package()))
    out = tmp_path/('export' if flag else 'plain')
    command = ['/tmp/dranppr_backend_build/bin/native-joint-audit', str(source), str(out), 'central_lm', '1']
    if flag:
        command.append('--linearization')
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    return out


def test_native_export_preserves_solver_and_physical_linear_model(tmp_path):
    out = exported(tmp_path)
    plain = exported(tmp_path, False)
    assert (out/'final_estimates.csv').read_bytes() == (plain/'final_estimates.csv').read_bytes()
    from experiments.network_noise_support import assemble
    data = json.loads((out/'linearization.json').read_text())
    a, b, groups = assemble(data)
    s = json.loads((out/'summary.json').read_text())
    assert .5*b@b == pytest.approx(s['final_cost'], abs=1e-11)
    assert np.linalg.norm(a.T@b) == pytest.approx(s['posthoc_gradient_norm'], abs=1e-10)
    assert a.shape == (48, 21)
    assert np.array_equal(np.bincount(groups), [4, 4, 4, 18, 18])
    split, other, types = assemble(data, split=True)
    assert split.shape == (48, 27)
    assert np.array_equal(other, b) and np.array_equal(types, groups)
    # Splitting copies adds exactly one free relative SE2 gauge in this fixture.
    assert np.linalg.matrix_rank(split)-np.linalg.matrix_rank(a) == 3


def test_suffix_weights_preserve_prefix_and_add_variance_in_physical_units():
    from experiments.network_noise_support import apply_scales
    data = weighted()
    before = copy.deepcopy(data)
    updated = apply_scales(data, [2., 3., 4., 5., 6.], cutoff=1., added_variance=.0225)
    assert data == before
    for robot in updated['robots']:
        for f in robot['odometry']:
            scales = [2., 3., 4.] if f['target'] > 1 else [1., 1., 1.]
            assert f['sigmas'] == pytest.approx([s*math.sqrt(q*f['dt']) for s,q in zip([.1,.1,.05], scales)])
        for f in robot['observations']:
            assert f['sigmas'] == pytest.approx([math.sqrt(.05**2*5+.0225), math.sqrt(.1**2*6)] if f['pose']>1 else [.05,.1])
    with pytest.raises(ValueError):
        apply_scales(data, [1.,1.,1.,0.,1.], 1., 0.)


def test_variance_model_group_partition():
    from experiments.network_noise_support import components
    groups = np.array([0,1,2,3,4,3,4])
    d, base = components(groups, observation_only=False)
    assert np.array_equal(d, np.eye(5)[groups])
    assert np.array_equal(base, np.zeros(7))
    d, base = components(groups, observation_only=True)
    assert d.shape == (7,2)
    assert np.array_equal(base, [1,1,1,0,0,0,0])
    assert np.array_equal(d.sum(axis=1)+base, np.ones(7))


def test_sparse_physical_assembly_matches_dense_without_changing_keys(tmp_path):
    from experiments.network_noise_support import assemble
    from scipy import sparse
    data = json.loads((exported(tmp_path)/'linearization.json').read_text())
    for split in (False, True):
        a, b, groups = assemble(data, split=split)
        compressed, other, types = assemble(data, split=split, sparse_output=True)
        assert sparse.issparse(compressed)
        np.testing.assert_array_equal(compressed.toarray(), a)
        np.testing.assert_array_equal(other, b)
        np.testing.assert_array_equal(types, groups)


def test_sparse_fit_cli_uses_sparse_implementation(tmp_path):
    source = exported(tmp_path)/'linearization.json'
    target = tmp_path/'sparse_fit'
    result = subprocess.run([sys.executable, '-m', 'experiments.audit_network_noise_support',
        'fit', '--source', str(source), '--arm', 'joint_r', '--implementation', 'sparse',
        '--output', str(target)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    fit = json.loads((target/'fit.json').read_text())
    assert fit['implementation'] == 'sparse_square_root_reml_exact_blocked_traces'
    assert not (target/'contrast.npz').exists()
