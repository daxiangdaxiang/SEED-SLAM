import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from analyze_residual_evidence import analyze


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_pairs_variants_by_dataset_robot_and_time(tmp_path):
    synthetic = []
    for initial, iteration, naive in ((0., 0, 1.), (1., 0, 100.), (1., 10, 2.)):
        synthetic.append(dict(case='unchanged_data', seed=0,
                              initial_error_m=initial, iterations=iteration,
                              fitted_ml_scale2=naive, scale_squared=1.))
    synthetic += [
        dict(case='one_corrupt_observation', seed=0, initial_error_m=0,
             iterations=0, fitted_ml_scale2=8., scale_squared=5.),
        dict(case='unidentifiable_state_equivalent_bias', seed=0,
             initial_error_m=0, iterations=0, fitted_ml_scale2=40., scale_squared=1.),
    ]
    for initial, iteration, naive in ((0., 0, 2.), (1., 0, 200.), (1., 10, 4.)):
        synthetic.append(dict(case='unchanged_data', seed=1,
                              initial_error_m=initial, iterations=iteration,
                              fitted_ml_scale2=naive, scale_squared=2.))
    synthetic += [
        dict(case='one_corrupt_observation', seed=1, initial_error_m=0,
             iterations=0, fitted_ml_scale2=16., scale_squared=10.),
        dict(case='unidentifiable_state_equivalent_bias', seed=1,
             initial_error_m=0, iterations=0, fitted_ml_scale2=80., scale_squared=2.),
    ]
    write_csv(tmp_path / 'synthetic.csv', synthetic)
    native = []
    for variant, naive, contrast in (
            ('batch_refined', 1., 2.), ('incremental', 1.01, 2.02),
            ('state_perturbed_01', 100., 2.2), ('state_perturbed_10', 10000., 200.)):
        native.append(dict(dataset=1, robot=1, elapsed_s=10, variant=variant,
                           fitted_ml_scale2=naive, scale_squared=contrast))
    write_csv(tmp_path / 'dataset1_evidence.csv', native)
    predictive = []
    for seed in (0, 1):
        for case, target_score, other_score in (
                ('nominal', 1., 1.), ('state_equivalent_bias', 1., 1.),
                ('group_sensor_degradation', 9., 1.),
                ('one_wrong_relationship', 100., 1.)):
            predictive += [
                dict(case=case, seed=seed, block=0, group='object_sensor_0',
                     target=case == 'one_wrong_relationship', identifiable=True,
                     energy_per_dimension=target_score),
                dict(case=case, seed=seed, block=1, group='object_sensor_1',
                     target=False, identifiable=True,
                     energy_per_dimension=other_score),
            ]
    write_csv(tmp_path / 'predictive.csv', predictive)
    write_csv(tmp_path / 'dataset1_predictive.csv', [
        dict(dataset=1, robot=1, elapsed_s=10, factor_kind='observation',
             identifiable=True, energy_per_dimension=.5),
        dict(dataset=1, robot=1, elapsed_s=10, factor_kind='observation',
             identifiable=False, energy_per_dimension=''),
        dict(dataset=1, robot=1, elapsed_s=10, factor_kind='odometry',
             identifiable=True, energy_per_dimension=2.),
        dict(dataset=1, robot=1, elapsed_s=10, factor_kind='odometry',
             identifiable=False, energy_per_dimension=''),
    ])
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(summaries=[dict(
        dataset=1, updates=5, p50_ms=.2, p95_ms=.4, p99_ms=.8,
        max_ms=1., max_serial_backlog_s=.01)])))

    result = analyze(tmp_path)

    assert result['synthetic']['naive_initialization_inflation'] == pytest.approx(100.)
    assert result['synthetic']['contrast_initialization_spread'] == pytest.approx(0.)
    assert result['synthetic']['localized_corruption_contrast_ratio'] == pytest.approx(5.)
    assert result['synthetic']['state_equivalent_bias_contrast_ratio'] == pytest.approx(1.)
    assert result['native']['small_perturbation']['naive_median_ratio'] == pytest.approx(100.)
    assert result['native']['small_perturbation']['contrast_median_ratio'] == pytest.approx(1.1)
    assert result['native']['incremental']['contrast_max_relative_difference'] == pytest.approx(.01)
    assert result['predictive']['state_equivalent_max_absolute_difference'] == pytest.approx(0.)
    assert result['predictive']['degraded_group_median_ratio'] == pytest.approx(9.)
    assert result['predictive']['degraded_unaffected_median_ratio'] == pytest.approx(1.)
    assert result['predictive']['wrong_relationship_target_to_other_median'] == pytest.approx(100.)
    assert result['native_predictive']['observation']['identifiable_fraction'] == pytest.approx(.5)
    assert result['native_predictive']['observation']['median_energy_per_dimension'] == pytest.approx(.5)
    assert result['native_predictive']['odometry']['median_energy_per_dimension'] == pytest.approx(2.)
    assert result['timing']['maximum_p99_ms'] == pytest.approx(.8)
