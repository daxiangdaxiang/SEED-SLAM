"""Controlled comparison of naive, predictive, oracle, and relation-edit arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_adaptation import (
    candidate_predictive_nll,
    gnc_geman_mcclure_fit,
    huber_state_fit,
    naive_scale_from_state,
    predictive_group_scale,
    weighted_state_fit,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_problem(rng, scale, dimension=5, references=25, sensors=50):
    truth = rng.normal(size=dimension)
    reference_a = rng.normal(size=(references, dimension))
    sensor_a = rng.normal(size=(sensors, dimension))
    reference_y = reference_a @ truth + rng.normal(size=references)
    sensor_y = sensor_a @ truth + np.sqrt(scale) * rng.normal(size=sensors)
    return truth, reference_a, reference_y, sensor_a, sensor_y


def state_rmse(truth, reference_a, reference_y, sensor_a, sensor_y, scale):
    design = np.vstack([reference_a, sensor_a])
    measurement = np.concatenate([reference_y, sensor_y])
    variance = np.concatenate([np.ones(len(reference_y)),
                               np.full(len(sensor_y), scale)])
    state = weighted_state_fit(design, measurement, variance).state
    return float(np.sqrt(np.mean((state - truth)**2)))


def scale_case(rng, true_scale):
    truth, reference_a, reference_y, sensor_a, sensor_y = draw_problem(
        rng, true_scale)
    bad_state = truth + 10.
    naive = naive_scale_from_state(sensor_a, sensor_y, bad_state)
    predictive = predictive_group_scale(
        reference_a, reference_y, sensor_a, sensor_y)
    design = np.vstack([reference_a, sensor_a])
    measurement = np.concatenate([reference_y, sensor_y])
    nominal_variance = np.ones(len(measurement))
    huber = huber_state_fit(design, measurement, nominal_variance)
    gnc_gm = gnc_geman_mcclure_fit(
        design, measurement, nominal_variance)
    sensor_slice = slice(len(reference_y), None)
    return dict(
        true_scale=true_scale, naive_scale=naive, predictive_scale=predictive,
        fixed_rmse=state_rmse(truth, reference_a, reference_y,
                              sensor_a, sensor_y, 1.),
        naive_rmse=state_rmse(truth, reference_a, reference_y,
                              sensor_a, sensor_y, naive),
        predictive_rmse=state_rmse(truth, reference_a, reference_y,
                                   sensor_a, sensor_y, predictive),
        huber_rmse=float(np.sqrt(np.mean((huber.state - truth)**2))),
        gnc_gm_rmse=float(np.sqrt(np.mean((gnc_gm.state - truth)**2))),
        huber_sensor_weight=float(np.mean(huber.weights[sensor_slice])),
        gnc_gm_sensor_weight=float(np.mean(gnc_gm.weights[sensor_slice])),
        oracle_rmse=state_rmse(truth, reference_a, reference_y,
                               sensor_a, sensor_y, true_scale))


def relation_case(rng):
    truth, reference_a, reference_y, peers_a, peers_y = draw_problem(
        rng, 1., sensors=49)
    direction = truth / max(np.linalg.norm(truth), 1e-12)
    true_row = 5. * direction
    wrong_row = -true_row
    target = float(true_row @ truth + rng.normal())
    peer_scale = predictive_group_scale(
        reference_a, reference_y, peers_a, peers_y)
    scores = candidate_predictive_nll(
        reference_a, reference_y, peers_a, peers_y,
        np.vstack([true_row, wrong_row]), target, peer_scale)
    equivalent = candidate_predictive_nll(
        reference_a, reference_y, peers_a, peers_y,
        np.vstack([true_row, true_row]), target, peer_scale)
    wrong_a = np.vstack([peers_a, wrong_row])
    correct_a = np.vstack([peers_a, true_row])
    measurements = np.concatenate([peers_y, [target]])
    wrong_design = np.vstack([reference_a, wrong_a])
    wrong_values = np.concatenate([reference_y, measurements])
    nominal_variance = np.ones(len(wrong_values))
    huber = huber_state_fit(wrong_design, wrong_values, nominal_variance)
    gnc_gm = gnc_geman_mcclure_fit(
        wrong_design, wrong_values, nominal_variance)
    inclusive_scale = predictive_group_scale(
        reference_a, reference_y, wrong_a, measurements)
    return dict(
        peer_scale=peer_scale, inclusive_wrong_scale=inclusive_scale,
        true_candidate_nll=float(scores[0]), wrong_candidate_nll=float(scores[1]),
        chose_true=bool(scores[0] < scores[1]),
        equivalent_nll_difference=float(abs(equivalent[0] - equivalent[1])),
        fixed_wrong_rmse=state_rmse(
            truth, reference_a, reference_y, wrong_a, measurements, 1.),
        inflated_wrong_rmse=state_rmse(
            truth, reference_a, reference_y, wrong_a, measurements,
            inclusive_scale),
        huber_wrong_rmse=float(np.sqrt(np.mean((huber.state - truth)**2))),
        gnc_gm_wrong_rmse=float(np.sqrt(np.mean((gnc_gm.state - truth)**2))),
        huber_target_weight=float(huber.weights[-1]),
        gnc_gm_target_weight=float(gnc_gm.weights[-1]),
        edited_rmse=state_rmse(
            truth, reference_a, reference_y, correct_a, measurements,
            peer_scale),
        oracle_rmse=state_rmse(
            truth, reference_a, reference_y, correct_a, measurements, 1.))


def audit(seeds):
    rows = []
    for seed in range(seeds):
        nominal = scale_case(np.random.default_rng(1000 + seed), 1.)
        changed = scale_case(np.random.default_rng(2000 + seed), 9.)
        relation = relation_case(np.random.default_rng(3000 + seed))
        row = {'seed': seed}
        row.update({f'nominal_{key}': value for key, value in nominal.items()})
        row.update({f'sensor_{key}': value for key, value in changed.items()})
        row.update({f'relation_{key}': value for key, value in relation.items()})
        rows.append(row)
    return rows


def summarize(rows):
    field = lambda name: np.asarray([row[name] for row in rows], dtype=float)
    return dict(
        seeds=len(rows),
        nominal=dict(
            naive_scale_inflation=float(np.median(field('nominal_naive_scale'))),
            predictive_scale_ratio=float(np.median(field('nominal_predictive_scale'))),
            fixed_to_oracle_rmse=float(np.median(
                field('nominal_fixed_rmse') / field('nominal_oracle_rmse'))),
            naive_to_oracle_rmse=float(np.median(
                field('nominal_naive_rmse') / field('nominal_oracle_rmse'))),
            predictive_to_oracle_rmse=float(np.median(
                field('nominal_predictive_rmse') / field('nominal_oracle_rmse'))),
            huber_to_oracle_rmse=float(np.median(
                field('nominal_huber_rmse') / field('nominal_oracle_rmse'))),
            gnc_gm_to_oracle_rmse=float(np.median(
                field('nominal_gnc_gm_rmse') / field('nominal_oracle_rmse'))),
            huber_sensor_weight=float(np.median(
                field('nominal_huber_sensor_weight'))),
            gnc_gm_sensor_weight=float(np.median(
                field('nominal_gnc_gm_sensor_weight')))),
        sensor_change=dict(
            true_scale=9.,
            naive_scale_ratio=float(np.median(field('sensor_naive_scale'))),
            predictive_scale_ratio=float(np.median(field('sensor_predictive_scale'))),
            fixed_to_oracle_rmse=float(np.median(
                field('sensor_fixed_rmse') / field('sensor_oracle_rmse'))),
            naive_to_oracle_rmse=float(np.median(
                field('sensor_naive_rmse') / field('sensor_oracle_rmse'))),
            predictive_to_oracle_rmse=float(np.median(
                field('sensor_predictive_rmse') / field('sensor_oracle_rmse'))),
            huber_to_oracle_rmse=float(np.median(
                field('sensor_huber_rmse') / field('sensor_oracle_rmse'))),
            gnc_gm_to_oracle_rmse=float(np.median(
                field('sensor_gnc_gm_rmse') / field('sensor_oracle_rmse'))),
            huber_sensor_weight=float(np.median(
                field('sensor_huber_sensor_weight'))),
            gnc_gm_sensor_weight=float(np.median(
                field('sensor_gnc_gm_sensor_weight')))),
        wrong_relation=dict(
            true_candidate_fraction=float(np.mean(field('relation_chose_true'))),
            inclusive_scale_inflation=float(np.median(
                field('relation_inclusive_wrong_scale') /
                field('relation_peer_scale'))),
            edited_to_oracle_rmse=float(np.median(
                field('relation_edited_rmse') / field('relation_oracle_rmse'))),
            fixed_wrong_to_oracle_rmse=float(np.median(
                field('relation_fixed_wrong_rmse') / field('relation_oracle_rmse'))),
            inflated_wrong_to_oracle_rmse=float(np.median(
                field('relation_inflated_wrong_rmse') /
                field('relation_oracle_rmse'))),
            huber_wrong_to_oracle_rmse=float(np.median(
                field('relation_huber_wrong_rmse') /
                field('relation_oracle_rmse'))),
            gnc_gm_wrong_to_oracle_rmse=float(np.median(
                field('relation_gnc_gm_wrong_rmse') /
                field('relation_oracle_rmse'))),
            huber_target_weight=float(np.median(
                field('relation_huber_target_weight'))),
            gnc_gm_target_weight=float(np.median(
                field('relation_gnc_gm_target_weight'))),
            equivalent_candidate_nll_difference=float(np.max(
                field('relation_equivalent_nll_difference')))))


def report(summary):
    return f'''# Cause-Separated Adaptation Audit

Across `{summary['seeds']}` controlled linear-Gaussian seeds, a poor current
state inflated the naive sensor variance by
`{summary['nominal']['naive_scale_inflation']:.1f}x`. Factor-excluded predictive
calibration estimated `{summary['nominal']['predictive_scale_ratio']:.2f}x` and
its state RMSE was `{summary['nominal']['predictive_to_oracle_rmse']:.3f}x` the
oracle arm. The fixed, naive, and predictive state-RMSE ratios were
`{summary['nominal']['fixed_to_oracle_rmse']:.3f}x`,
`{summary['nominal']['naive_to_oracle_rmse']:.3f}x`, and
`{summary['nominal']['predictive_to_oracle_rmse']:.3f}x`, respectively.

For a true `9x` sensor variance, predictive calibration estimated
`{summary['sensor_change']['predictive_scale_ratio']:.2f}x`, while naive
calibration from the deliberately inaccurate state estimated
`{summary['sensor_change']['naive_scale_ratio']:.2f}x`. Fixed, naive, and
predictive state-RMSE ratios were
`{summary['sensor_change']['fixed_to_oracle_rmse']:.3f}x`,
`{summary['sensor_change']['naive_to_oracle_rmse']:.3f}x`, and
`{summary['sensor_change']['predictive_to_oracle_rmse']:.3f}x` oracle.

With one wrong relation and one independently supported alternative, predictive
likelihood selected the correct candidate in
`{summary['wrong_relation']['true_candidate_fraction']:.3f}` of seeds. Fitting
one inclusive sensor scale instead inflated that group by
`{summary['wrong_relation']['inclusive_scale_inflation']:.2f}x`; it did not
repair the relation, with state-RMSE ratio
`{summary['wrong_relation']['inflated_wrong_to_oracle_rmse']:.3f}x` versus
`{summary['wrong_relation']['fixed_wrong_to_oracle_rmse']:.3f}x` for the fixed
wrong relation and `{summary['wrong_relation']['edited_to_oracle_rmse']:.3f}x`
after the correct edit. Duplicate,
observationally equivalent candidates differed by at most
`{summary['wrong_relation']['equivalent_candidate_nll_difference']:.2e}` NLL,
so the method correctly provides no identity evidence in that case.

The robust controls use fixed conventional settings rather than learned sensor
covariances: Huber IRLS uses `delta=1.345`, and the clearly labelled GNC-style
Geman--McClure approximation uses a `2 sigma` terminal cutoff. Under nominal
Gaussian noise their state-RMSE ratios were
`{summary['nominal']['huber_to_oracle_rmse']:.3f}x` and
`{summary['nominal']['gnc_gm_to_oracle_rmse']:.3f}x`. Under the true `9x`
sensor variance they retained median sensor weights
`{summary['sensor_change']['huber_sensor_weight']:.3f}` and
`{summary['sensor_change']['gnc_gm_sensor_weight']:.3f}`, with RMSE ratios
`{summary['sensor_change']['huber_to_oracle_rmse']:.3f}x` and
`{summary['sensor_change']['gnc_gm_to_oracle_rmse']:.3f}x`. For the isolated
wrong relation they assigned target weights
`{summary['wrong_relation']['huber_target_weight']:.3f}` and
`{summary['wrong_relation']['gnc_gm_target_weight']:.3f}`, reaching
`{summary['wrong_relation']['huber_wrong_to_oracle_rmse']:.3f}x` and
`{summary['wrong_relation']['gnc_gm_wrong_to_oracle_rmse']:.3f}x` oracle RMSE.
These controls test behavior, not official riMESA, GNC, or AEROS code.

This audit verifies mechanisms on a small exact linear model. It is neither a
native-noise claim nor a distributed DRANPPR comparison.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    producer = Path(__file__).resolve()
    model = producer.parents[1] / 'scripts' / 'dranppr_adaptation.py'
    source_hashes = {str(path.relative_to(producer.parents[1])): sha256(path)
                     for path in (producer, model)}
    rows = audit(args.seeds)
    summary = summarize(rows)
    with (args.output / 'rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report(summary))
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source_hashes=source_hashes, synthetic=True, truth_used=True,
        argv=sys.argv), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
