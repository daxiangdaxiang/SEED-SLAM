import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_pose3_physical_jacobians_and_sparse_calibration_match_dense(tmp_path):
    binary = ROOT/'build/native/bin/native-pose3-calibration-audit'
    assert binary.is_file(), 'missing SE3 physical diagnostic'
    completed = subprocess.run([str(binary), str(tmp_path/'run'), '31', 'heterogeneous'],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    data = json.loads((tmp_path/'run/training.json').read_text())
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['jacobian_max_error'] < 1e-6
    assert summary['root_displacement'] < 1e-12
    assert summary['training_test_measurements_differ']
    assert summary['pose_tangent_dimension'] == 6
    assert summary['object_tangent_dimension'] == 6
    assert min(summary['truth_translation_extent']) > 1.
    assert min(summary['truth_rotation_extent']) > .1
    from dranppr_variance_components import ContrastVarianceModel
    from dranppr_sparse_variance_components import SparseVarianceModel
    a, b = np.asarray(data['a']), np.asarray(data['b'])
    d = np.eye(4)[data['groups']]
    dense = ContrastVarianceModel(a, b, d, np.zeros(len(b)))
    sparse = SparseVarianceModel(a, b, d, np.zeros(len(b)))
    theta = np.log([.2, 3., 5., .5])
    np.testing.assert_allclose(sparse.nll(theta), dense.nll(theta), rtol=1e-10)
    np.testing.assert_allclose(sparse.gradient(theta), dense.gradient(theta), atol=1e-7)


def test_pose3_weight_intervention_keeps_test_measurements_and_initialization(tmp_path):
    binary = ROOT/'build/native/bin/native-pose3-calibration-audit'
    for arm, scales in [('fixed', []), ('changed', ['.1','4','9','.25'])]:
        result = subprocess.run([str(binary),str(tmp_path/arm),'32','heterogeneous',*scales],
                                capture_output=True,text=True,timeout=60)
        assert result.returncode == 0, result.stderr
    assert (tmp_path/'fixed/test_input.json').read_bytes() == (tmp_path/'changed/test_input.json').read_bytes()
    assert (tmp_path/'fixed/training.json').read_bytes() == (tmp_path/'changed/training.json').read_bytes()
    fixed = json.loads((tmp_path/'fixed/summary.json').read_text())
    changed = json.loads((tmp_path/'changed/summary.json').read_text())
    assert fixed['used_scales'] == [1.]*4
    assert changed['used_scales'] == [.1,4.,9.,.25]
