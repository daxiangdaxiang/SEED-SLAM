import pytest

from experiments.analyze_pose3_calibration import paired_summary


def test_paired_summary_counts_regressions_and_uses_ratio_of_means():
    result = paired_summary([1.,3.,2.], [2.,1.,2.], [1.,1.,1.])
    assert result['calibrated_wins'] == 1
    assert result['calibrated_losses'] == 1
    assert result['ties'] == 1
    assert result['mean_reduction_percent'] == pytest.approx(100./6.)
    assert result['max_paired_increase'] == 1.


@pytest.mark.parametrize('other', [[], [1.], [1.,float('nan')], [1.,-1.]])
def test_paired_summary_rejects_invalid_support(other):
    with pytest.raises(ValueError):
        paired_summary([1.,2.],other,[1.,2.])
