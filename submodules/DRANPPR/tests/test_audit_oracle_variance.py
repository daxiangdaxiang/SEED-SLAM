import pytest

from experiments import audit_oracle_variance as audit


def test_fixed_abba_variance_protocol_has_no_fitted_parameter():
    assert audit.arms() == [('nominal1', 0.), ('added1', .0225),
                            ('added2', .0225), ('nominal2', 0.)]
    assert audit.cases() == [('clean_lazy', 'd2_clean', True),
                             ('gaussian_lazy', 'd2_gaussian', True),
                             ('gaussian_ordinary', 'd2_gaussian', False)]


def summary():
    return dict(frames=4, packets=8, physical_factors=30, prefix_physical_factors=22,
                initial_cost=12., physical_cost=15.)


def test_problem_contract_allows_distinct_weighted_losses_but_not_prefix_changes():
    a, b = summary(), summary()
    b['physical_cost'] = 2.
    audit.verify_problem(a, b)
    b['initial_cost'] += .01
    with pytest.raises(ValueError, match='initial_cost'):
        audit.verify_problem(a, b)


@pytest.mark.parametrize('key', ['frames', 'packets', 'physical_factors', 'prefix_physical_factors'])
def test_problem_contract_rejects_changed_information(key):
    a, b = summary(), summary()
    b[key] += 1
    with pytest.raises(ValueError, match=key):
        audit.verify_problem(a, b)
