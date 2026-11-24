import importlib.util
import math
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'experiments/audit_observation_pair_evidence.py'
    assert path.is_file(), 'pair evidence diagnostic not implemented'
    spec = importlib.util.spec_from_file_location('pair_evidence', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def observation(subject, distance, angle):
    return dict(landmark_subject=subject, range_m=distance, bearing_rad=angle)


def test_pair_distance_is_rigid_frame_invariant_and_variance_matches_derivatives():
    m = module()
    a, b = observation(6, 2., .2), observation(7, 3., -.3)
    distance, variance = m.distance_variance(a, b)
    theta, tx, ty = .7, .4, -.2
    changed = []
    for row in [a, b]:
        x = row['range_m']*math.cos(row['bearing_rad']+theta)+tx
        y = row['range_m']*math.sin(row['bearing_rad']+theta)+ty
        changed.append(observation(row['landmark_subject'], math.hypot(x,y), math.atan2(y,x)))
    assert m.distance_variance(*changed)[0] == pytest.approx(distance, abs=1e-12)
    finite_variance = 0.
    for index in [0, 1]:
        for key, sigma in [('range_m', .1), ('bearing_rad', .05)]:
            plus, minus = [dict(a), dict(b)], [dict(a), dict(b)]
            plus[index][key] += 1e-6
            minus[index][key] -= 1e-6
            derivative = (m.distance_variance(*plus)[0]-m.distance_variance(*minus)[0])/2e-6
            finite_variance += sigma*sigma*derivative*derivative
    assert variance == pytest.approx(finite_variance, rel=1e-9)


def test_only_previous_pair_measurement_enters_prediction():
    m = module()
    frames = []
    for t in range(3):
        frames.append([dict(observation(6, 2., 0.), source_row=2*t, timestamp_s=float(t)),
                       dict(observation(7, 3.+.1*t, .5), source_row=2*t+1, timestamp_s=float(t))])
    prefix = m.pair_evidence(frames[:2])
    full = m.pair_evidence(frames)
    frames[-1][-1]['range_m'] = 100.
    changed = m.pair_evidence(frames)
    assert prefix == full[:1] == changed[:1]
    assert full[0]['previous_timestamp_s'] == 0.
    assert full[0]['timestamp_s'] == 1.


def test_degenerate_pair_has_no_linearized_distance_evidence():
    with pytest.raises(ValueError, match='degenerate'):
        module().distance_variance(observation(6, 2., 0.), observation(7, 2., 0.))
