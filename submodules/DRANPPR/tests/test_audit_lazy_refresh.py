import pytest
from pathlib import Path

from experiments import audit_lazy_refresh as audit
from experiments.audit_lazy_refresh import audit_updates, replay_arms


def records():
    return [dict(cost_before='10', cost_after='9', accepted='1', pcg_iterations='1',
                 full_step_prediction='2', full_step_decrease='1', refresh_requests='0',
                 refresh_rounds='0', optimizer_bytes='100', control_bytes='200'),
            dict(cost_before='11', cost_after='10.5', accepted='1', pcg_iterations='1',
                 full_step_prediction='2', full_step_decrease='-1', refresh_requests='1',
                 refresh_rounds='0', optimizer_bytes='150', control_bytes='200'),
            dict(cost_before='12', cost_after='10', accepted='1', pcg_iterations='1',
                 full_step_prediction='2', full_step_decrease='2', refresh_requests='0',
                 refresh_rounds='1', optimizer_bytes='100', control_bytes='200')]


def test_audit_checks_deferred_refresh_and_counts_all_bytes():
    result = audit_updates(records(), True, False)
    assert result['application_bytes'] == 950
    assert result['refresh_rounds'] == result['refresh_requests'] == 1
    assert result['full_step_failures'] == 1
    assert result['accepted_frames'] == 3
    assert result['longest_no_refresh_frames'] == 2
    assert result['full_step_gain_ratio_median'] == pytest.approx(.5)


@pytest.mark.parametrize('field,value,message', [
    ('refresh_rounds', '0', 'deferred refresh'),
    ('cost_after', '13', 'physical descent'),
    ('pcg_iterations', '2', 'iteration cap'),
    ('refresh_requests', '1', 'full-step decision')])
def test_audit_rejects_invalid_policies_or_descent(field, value, message):
    rows = records()
    rows[-1][field] = value
    with pytest.raises(ValueError, match=message):
        audit_updates(rows, True, False)


def test_audit_does_not_hide_terminal_pending_refresh_or_nonfinite_cost():
    rows = records()[:2]
    assert audit_updates(rows, True, True)['refresh_requests'] == 1
    with pytest.raises(ValueError, match='pending'):
        audit_updates(rows, True, False)
    rows[0]['cost_before'] = 'nan'
    with pytest.raises(ValueError, match='finite'):
        audit_updates(rows, True, True)


def test_normal_only_protocol_changes_only_refresh_policy_in_abba_order():
    assert replay_arms(True) == [
        ('reference1', False, False), ('candidate1', False, True),
        ('candidate2', False, True), ('reference2', False, False)]


def test_original_protocol_retains_conditional_and_normal_controls():
    assert replay_arms(False) == [
        ('reference1', False, False), ('candidate1', True, True),
        ('candidate2', True, True), ('reference2', False, False),
        ('conditional_reference', True, False), ('lazy_normal', False, True)]


def test_explicit_inputs_do_not_reuse_development_defaults():
    assert audit.select_inputs(Path('/repo'), 2, Path('/events'), Path('/warm')) == (
        Path('/events'), Path('/warm'))
    events, warm = audit.select_inputs(Path('/repo'), 4, None, None)
    assert events.name == 'd4_events.jsonl'
    assert warm.parent.name == 'd4_one'


@pytest.mark.parametrize('dataset,events,warm', [(2, None, None),
    (4, Path('/events'), None), (4, None, Path('/warm'))])
def test_explicit_inputs_must_be_paired(dataset, events, warm):
    with pytest.raises(ValueError):
        audit.select_inputs(Path('/repo'), dataset, events, warm)
