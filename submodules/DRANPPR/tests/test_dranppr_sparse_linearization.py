from pathlib import Path
import sys

import gtsam
import numpy as np
import pytest
from scipy.sparse import isspmatrix_csc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))
from audit_residual_evidence import linearize


@pytest.mark.parametrize('zero_residual', [False, True])
def test_sparse_linearization_matches_explicit_ordering_and_whitening(zero_residual):
    from dranppr_sparse_linearization import sparse_linearize
    x, y, l = gtsam.symbol('x', 9), gtsam.symbol('x', 2), gtsam.symbol('l', 17)
    values = gtsam.Values()
    values.insert(x, gtsam.Pose2(.3, .5, .2))
    values.insert(y, gtsam.Pose2(.1, -.1, -.2))
    values.insert(l, np.array([2., 1.]))
    graph = gtsam.NonlinearFactorGraph()
    relative = values.atPose2(x).between(values.atPose2(y)) if zero_residual else gtsam.Pose2()
    graph.add(gtsam.BetweenFactorPose2(x, y, relative,
              gtsam.noiseModel.Diagonal.Sigmas(np.array([.01, .2, .3]))))
    pose, point = values.atPose2(y), values.atPoint2(l)
    graph.add(gtsam.BearingRangeFactor2D(y, l, pose.bearing(point), pose.range(point),
              gtsam.noiseModel.Diagonal.Sigmas(np.array([.05, .1]))))
    # The values container may have variables absent from this graph.
    values.insert(gtsam.symbol('x', 99), gtsam.Pose2())
    actual, residual, keys = sparse_linearize(graph, values)
    expected, expected_residual, expected_keys = linearize(graph, values)
    assert isspmatrix_csc(actual)
    assert keys == expected_keys
    np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(residual, expected_residual, rtol=1e-13, atol=1e-12)


def test_sparse_linearization_retains_zero_rows_and_columns():
    from dranppr_sparse_linearization import sparse_linearize
    x = gtsam.symbol('x', 0)
    values = gtsam.Values()
    values.insert(x, gtsam.Pose2())
    def error(factor, values, jacobians):
        if jacobians is not None:
            jacobians[0] = np.zeros((2, 3))
        return np.zeros(2)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(2), [x], error))
    a, residual, keys = sparse_linearize(graph, values)
    assert a.shape == (2, 3)
    assert a.nnz == 0
    np.testing.assert_array_equal(residual, np.zeros(2))
    assert keys == [x]


def test_sparse_linearization_rejects_empty_physical_graph():
    from dranppr_sparse_linearization import sparse_linearize
    with pytest.raises(ValueError, match='no variables'):
        sparse_linearize(gtsam.NonlinearFactorGraph(), gtsam.Values())
