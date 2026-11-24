from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments'))


def test_variance_profile_refits_nuisance_scale_and_preserves_fixed_component():
    from audit_variance_profiles import profile_component
    from dranppr_variance_components import ContrastVarianceModel
    rng = np.random.default_rng(242)
    groups = np.repeat([0, 1], 20)
    design = np.column_stack([groups == 0, groups == 1]).astype(float)
    y = rng.normal(size=40) * np.where(groups == 0, .5, 2.)
    model = ContrastVarianceModel(design, y, design, np.zeros(40))
    fit = model.fit(np.zeros(2))
    profiled = profile_component(model, 0, 1.5, np.log(fit['components']))
    assert profiled['success']
    assert profiled['components'][0] == pytest.approx(1.5)
    assert profiled['components'][1] == pytest.approx(np.var(y[groups == 1], ddof=1), rel=1e-6)
    assert profiled['nll'] > fit['nll']
    assert profiled['nll'] == pytest.approx(model.nll(np.log(profiled['components'])))


def test_single_component_profile_has_no_nuisance_optimizer():
    from audit_variance_profiles import profile_component
    from dranppr_variance_components import ContrastVarianceModel
    model = ContrastVarianceModel(np.ones((5, 1)), np.arange(5.), np.ones((5, 1)), np.zeros(5))
    result = profile_component(model, 0, .4, np.zeros(1))
    assert result['success']
    assert result['components'] == [.4]
    assert result['nll'] == pytest.approx(model.nll(np.log([.4])))


def test_variance_profile_rejects_nonpositive_component():
    from audit_variance_profiles import profile_component
    with pytest.raises(ValueError, match='positive'):
        profile_component(None, 0, 0., np.zeros(1))
