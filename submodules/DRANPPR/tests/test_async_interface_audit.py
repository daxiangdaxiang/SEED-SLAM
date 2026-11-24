import json
from pathlib import Path

import numpy as np
import pytest

from experiments.async_interface_model import physical_factors, pose3_factors
from experiments.audit_async_interface import cases, full_reference, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_real_schur_matches_independent_full_system():
    path = ROOT/cases()['d2']
    if not path.exists():
        pytest.skip('archived MRCLAM fixture unavailable')
    data = json.loads(path.read_text())
    model, metadata = physical_factors(data)
    parity = full_reference(data, model)
    assert parity['boundary_absolute_error'] < 1e-8
    assert parity['full_relative_residual'] < 1e-7
    assert metadata['local_private_max_relative_residual'] < 1e-10
    assert len(model.indices) >= 2


def test_six_dof_fixture_matches_full_least_squares():
    path = ROOT/cases()['se3_31']
    if not path.exists():
        pytest.skip('archived SE3 fixture unavailable')
    model, metadata = pose3_factors(json.loads(path.read_text()))
    assert model.b.shape == (24, 24)
    assert metadata['full_solve_relative_error'] < 1e-6
    assert metadata['local_private_max_relative_residual'] < 1e-10


def test_summary_does_not_count_partial_horizons_as_successes():
    rows = [dict(case='x', scenario='delay', mode=mode, seed=31,
                 relative_gap=gap, relative_residual=.1, application_bytes=100,
                 bound_violations=0, horizon_complete=complete, finite=True)
            for mode,gap,complete in [('synchronous',.1,True), ('stale',.02,False), ('bounded',.03,True)]]
    result = summarize(rows)
    assert result['complete_runs'] == 2
    assert result['bounded_vs_stale_complete_pairs'] == 0
    assert result['bounded_vs_synchronous_gap_wins'] == 1


def test_adapter_rejects_truth_assisted_or_changed_se3_layout():
    with pytest.raises(ValueError):
        physical_factors(dict(schema='dranppr_physical_linearization_v1', truth_used=True))
    with pytest.raises(ValueError):
        pose3_factors(dict(schema='dranppr_pose3_physical_calibration_v1',
                          a=np.zeros((6, 6)).tolist(), b=np.zeros(6).tolist()))
