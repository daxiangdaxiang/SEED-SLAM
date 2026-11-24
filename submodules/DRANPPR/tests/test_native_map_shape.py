import numpy as np
import pytest

from experiments.audit_native_map_shape import distances, compare_maps, neighbor_object_starts


def record():
    covariance = np.kron(np.ones((3, 3)), 2.*np.eye(2)) + .1*np.eye(6)
    return dict(object_ids=[7, 9, 12], means=[[0., 0.], [3., 0.], [0., 4.]],
                covariance=covariance.tolist())


def test_common_translation_uncertainty_cancels_in_distance():
    rows = distances(record())
    assert [r['distance_m'] for r in rows] == pytest.approx([3., 4., 5.])
    assert [r['variance_m2'] for r in rows] == pytest.approx([.2]*3)
    assert [r['block_diagonal_variance_m2'] for r in rows] == pytest.approx([4.2]*3)


def test_gauge_and_order_invariance():
    original = record()
    rotation = np.array([[.6, -.8], [.8, .6]])
    rotated = dict(original, means=(np.array(original['means'])@rotation.T+[10., -7.]).tolist(),
                   covariance=(np.kron(np.eye(3), rotation)@original['covariance']@
                               np.kron(np.eye(3), rotation.T)).tolist())
    for row in compare_maps(original, rotated):
        assert row['difference_m'] == pytest.approx(0., abs=1e-12)
        assert row['combined_variance_m2'] == pytest.approx(.4)
    permutation = [2, 0, 1]
    columns = np.array([[2*i, 2*i+1] for i in permutation]).ravel()
    reordered = dict(object_ids=[original['object_ids'][i] for i in permutation],
        means=[original['means'][i] for i in permutation],
        covariance=np.array(original['covariance'])[np.ix_(columns, columns)].tolist())
    assert distances(reordered) == distances(original)


def test_uncertainty_combines_independent_robot_maps():
    changed = record()
    changed['means'][1][0] = 4.
    row = compare_maps(record(), changed)[0]
    assert row['difference_m'] == -1.
    assert row['absolute_standardized_difference'] == pytest.approx(1./np.sqrt(.4))


def test_coincident_points_and_invalid_covariance_are_explicit():
    bad = record()
    bad['means'][1] = bad['means'][0]
    assert distances(bad)[0]['status'] == 'undefined_distance_derivative'
    assert compare_maps(record(), bad)[0]['status'] == 'undefined_distance_derivative'
    for covariance in [np.zeros((5, 5)), -np.eye(6), np.full((6, 6), np.nan)]:
        with pytest.raises(ValueError, match='covariance'):
            distances(dict(record(), covariance=covariance.tolist()))
    with pytest.raises(ValueError, match='object'):
        distances(dict(record(), object_ids=[7, 7, 12]))


def test_neighbor_start_changes_only_object_values_with_matched_opposite_control():
    source = record()
    rows = [dict(robot=1, kind='pose', index=0, x=0., y=0., yaw=0., timestamp_s=0.)]
    rows += [dict(robot=1, kind='landmark', index=k, x=p[0], y=p[1])
             for k, p in zip(source['object_ids'], source['means'])]
    source['means'][1][0] = 4.
    starts, info = neighbor_object_starts(rows, source)
    assert info['common_objects'] == [7, 9, 12]
    assert info['object_delta_norm_m'] > .1
    assert starts['neighbor'][0] == starts['opposite'][0] == rows[0]
    for old, plus, minus in zip(rows[1:], starts['neighbor'][1:], starts['opposite'][1:]):
        assert (plus['x']+minus['x'])/2 == pytest.approx(old['x'])
        assert (plus['y']+minus['y'])/2 == pytest.approx(old['y'])
        assert old['index'] == plus['index'] == minus['index']
