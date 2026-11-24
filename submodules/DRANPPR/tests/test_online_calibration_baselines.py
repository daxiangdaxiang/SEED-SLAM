import math
import json

import pytest


def test_common_credits_use_smaller_causal_total_not_final_budget():
    from experiments.audit_online_calibration_baselines import common_credits
    a = [dict(robot=1, timestamp_s=3., optimizer_bytes=10, control_bytes=2),
         dict(robot=2, timestamp_s=4., optimizer_bytes=10, control_bytes=2)]
    b = [dict(a[0], optimizer_bytes=15), dict(a[1], optimizer_bytes=1)]
    assert common_credits(a, b) == [dict(robot=1, timestamp_s=3., application_bytes=12),
                                  dict(robot=2, timestamp_s=4., application_bytes=20)]


def test_common_credits_reject_missing_reordered_or_negative_messages():
    from experiments.audit_online_calibration_baselines import common_credits
    rows = [dict(robot=1, timestamp_s=3., optimizer_bytes=10, control_bytes=2),
            dict(robot=2, timestamp_s=4., optimizer_bytes=10, control_bytes=2)]
    for other in [rows[:1], rows[::-1], [dict(rows[0], optimizer_bytes=-1), rows[1]]]:
        with pytest.raises(ValueError):
            common_credits(rows, other)


def test_declared_dataset_durations_preserve_legacy_and_expand_without_retuning():
    from experiments.audit_online_calibration_baselines import benchmark_cases
    assert benchmark_cases({}) == [(1,120.),(4,120.)]
    assert benchmark_cases({'benchmark_cases':[{'dataset':2,'duration':120},
        {'dataset':3,'duration':120},{'dataset':9,'duration':300}]}) == [(2,120.),(3,120.),(9,300.)]


@pytest.mark.parametrize('cases', [[],[{'dataset':2}],
    [{'dataset':2,'duration':120},{'dataset':2,'duration':300}],
    [{'dataset':2,'duration':math.nan}],[{'dataset':2,'duration':-1}],
    [{'dataset':2.5,'duration':120}],[{'dataset':0,'duration':120}]])
def test_invalid_benchmark_case_contract_is_rejected(cases):
    from experiments.audit_online_calibration_baselines import benchmark_cases
    with pytest.raises(ValueError):
        benchmark_cases({'benchmark_cases':cases})


def test_missing_calibration_is_only_reportable_with_an_explicit_failed_fit(tmp_path):
    from experiments.audit_online_calibration_baselines import calibration_for_case
    with pytest.raises(ValueError):
        calibration_for_case(tmp_path,3,{'runs':[]})
    fit, reason = calibration_for_case(tmp_path,3,{'runs':[{'tag':'d3_fit','exit_code':'timeout_600s'}]})
    assert fit is None
    assert 'timeout_600s' in reason
    folder = tmp_path/'d3/fits/joint_qr'
    folder.mkdir(parents=True)
    (folder/'fit.json').write_text(json.dumps({'success':True,'scales':[1]*5,'truth_used':False}))
    fit, reason = calibration_for_case(tmp_path,3,{'runs':[]})
    assert fit['scales'] == [1]*5 and reason is None
    fit, reason = calibration_for_case(tmp_path,3,{'runs':[{'tag':'d3_fit','exit_code':'timeout_600s'}]})
    assert fit is None and reason == 'timeout_600s'


def test_dataset_subset_keeps_declared_duration_and_rejects_missing_or_duplicate_cases():
    from experiments.audit_online_calibration_baselines import benchmark_cases
    manifest = {'benchmark_cases':[{'dataset':2,'duration':120},{'dataset':9,'duration':300}]}
    assert benchmark_cases(manifest, [9]) == [(9,300.)]
    for selected in ([], [3], [9,9]):
        with pytest.raises(ValueError):
            benchmark_cases(manifest, selected)

