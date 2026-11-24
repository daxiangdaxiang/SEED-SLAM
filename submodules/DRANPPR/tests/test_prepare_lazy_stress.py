import json
import math

import pytest

from experiments import prepare_lazy_stress as stress


def lines():
    return [json.dumps(dict(kind='observation' if k % 2 else 'pose',
                           timestamp_s=float(k), bearing_rad=3.1, range_m=2.,
                           robot=1, source_row=k, landmark_subject=7))+'\n'
            for k in range(200)]


@pytest.mark.parametrize('kind', ['gaussian_bearing', 'sparse_bearing'])
def test_stress_preserves_prefix_future_and_nonbearing_fields(kind):
    original = lines()
    changed, manifest = stress.perturb(original, 10., 190., kind, 20261003)
    assert changed[:11] == original[:11]
    assert changed[191:] == original[191:]
    assert manifest['eligible_observations'] == 90
    assert manifest['changes']
    for old, new in zip(map(json.loads, original), map(json.loads, changed)):
        a, b = old.pop('bearing_rad'), new.pop('bearing_rad')
        assert old == new
        if old['kind'] == 'pose':
            assert a == b
        assert -math.pi <= b <= math.pi
    for item in manifest['changes']:
        assert item['timestamp_s'] > 10.
        assert item['timestamp_s'] <= 190.
        if kind == 'sparse_bearing':
            assert abs(item['added_bearing_rad']) == .5
    assert stress.perturb(original, 10., 190., kind, 20261003) == (changed, manifest)


def test_gaussian_stress_changes_all_eligible_observations_and_is_causal():
    original = lines()
    changed, manifest = stress.perturb(original, 10., 190., 'gaussian_bearing', 3)
    shorter, _ = stress.perturb(original, 10., 100., 'gaussian_bearing', 3)
    assert len(manifest['changes']) == 90
    assert changed[:101] == shorter[:101]


@pytest.mark.parametrize('cutoff,stop,kind', [(10., 10., 'gaussian_bearing'),
    (float('nan'), 20., 'gaussian_bearing'), (10., 20., 'unknown')])
def test_stress_rejects_invalid_contract(cutoff, stop, kind):
    with pytest.raises(ValueError):
        stress.perturb(lines(), cutoff, stop, kind, 1)
