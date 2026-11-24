import pytest


def test_trace_review_rejects_physical_change_even_if_budget_is_lower():
    from experiments.summarize_subspace import compare_traces
    row = dict(robot='1',timestamp_s='2',cost_before='3',cost_after='2',accepted='1',
        refresh_requests='0',refresh_rounds='0',refreshed_keys='0',factors_linearized='0',
        full_step_prediction='1',full_step_decrease='1',pcg_iterations='1',
        optimizer_bytes='20',control_bytes='0',pcg_relative_residual='.5')
    other = dict(row,optimizer_bytes='10',pcg_relative_residual='.5000000000000001')
    result = compare_traces([row],[other])
    assert result['prefix_exceedances']==0
    assert result['max_residual_discrepancy']<1e-14
    other['cost_after']='2.1'
    with pytest.raises(ValueError,match='physical trace changed'):
        compare_traces([row],[other])
