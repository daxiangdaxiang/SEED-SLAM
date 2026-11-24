"""Compare sealed response runs with prior predictors without reranking tuning."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_neighbor_prediction import review_pair
from experiments.audit_response_controls import load_model
from experiments.review_async_interface import verify_ledger
from experiments.review_history_prediction import review


def aggregate(items):
    result = {}
    for label in sorted({x['prediction']+'_vs_'+x['reference'] for x in items}):
        selected = [x for x in items if x['prediction']+'_vs_'+x['reference'] == label]
        for scope, subset in (
                ('all', selected),
                ('nonideal', [x for x in selected if x['scenario'] != 'ideal']),
                ('mrclam', [x for x in selected if x['case'].startswith('d') and x['scenario'] != 'ideal']),
                ('se3', [x for x in selected if x['case'].startswith('se3') and x['scenario'] != 'ideal'])):
            if subset:
                result[label+':'+scope] = dict(
                    gap={key: sum(x['gap_result'] == key for x in subset) for key in ('win', 'tie', 'loss')},
                    median_gap_reduction=float(np.median([x['gap_reduction'] for x in subset])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, archive = verify_ledger(args.root), verify_ledger(args.archive)
    for key in ('cases', 'scenarios', 'seeds', 'ticks', 'solver', 'history_size', 'topology'):
        if manifest[key] != archive[key]:
            raise ValueError('archive protocol mismatch: '+key)
    args.output.mkdir(parents=True, exist_ok=False)
    checked = review(args.root, args.prior)
    write_json(args.output/'independent_review.json', checked)
    items, hits = [], []
    archive_parity = 0
    for case in manifest['cases']:
        model, _ = load_model(args.root, case)
        current_data = np.load(args.root/case/'quadratic.npz')
        old_data = np.load(args.archive/case/'quadratic.npz')
        for key in current_data.files:
            np.testing.assert_array_equal(current_data[key], old_data[key])
        initial = np.sin(np.arange(len(model.c))+1.)
        for net in manifest['scenarios']:
            for seed in manifest['seeds']:
                traces = {mode: np.load(args.archive/case/f'{net}_{seed}_{mode}.npz')
                          for mode in archive['predictions']}
                current = {mode: np.load(args.root/case/f'{net}_{seed}_{mode}.npz')
                           for mode in manifest['predictions']}
                np.testing.assert_array_equal(traces['hold']['states'], current['hold']['states'])
                archive_parity += 1
                traces['response_linear'] = current['response_linear']
                for mode in ('response_linear', 'response_calibrated'):
                    candidate = current[mode]
                    for other, reference in traces.items():
                        if other == mode:
                            continue
                        metrics = review_pair(model, initial, candidate, reference)
                        items.append(dict(metrics, case=case, scenario=net, seed=seed, prediction=mode,
                            reference=other, gap_reduction=1.-metrics['candidate_gap']/max(metrics['reference_gap'], 1e-30)))
                        a = np.flatnonzero(candidate['relative_gaps'] < 1e-6)
                        b = np.flatnonzero(reference['relative_gaps'] < 1e-6)
                        if net != 'ideal' and case != 'coupled_spd' and len(a) and len(b):
                            hits.append(dict(case=case, scenario=net, seed=seed, prediction=mode, reference=other,
                                candidate_tick=int(a[0]), reference_tick=int(b[0]),
                                bytes_reduction=1.-float(candidate['counters'][a[0], 0])/reference['counters'][b[0], 0]))
    write_json(args.output/'comparisons.json', items)
    write_json(args.output/'target_hits.json', hits)
    summary = dict(comparisons=aggregate(items), archive_hold_parity=archive_parity,
        prefix_comparisons=len(items), all_prefix_budgets_equal=all(x['prefix_budget_equal'] for x in items),
        independent_metric_checks=checked['metric_checks'],
        max_normalized_bound_violation=checked['max_normalized_bound_violation'], cases=checked['cases'])
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'manifest.json', dict(all_terminal=True, root=str(args.root.resolve()),
        root_ledger=digest(args.root/'SHA256SUMS'), archive=str(args.archive.resolve()),
        archive_ledger=digest(args.archive/'SHA256SUMS'),
        scope='Fixed-model validation; retrospective error calibration does not establish stability'))
    for name in ('experiments/review_response_prediction.py', 'experiments/review_history_prediction.py',
                 'experiments/audit_neighbor_prediction.py', 'experiments/audit_response_controls.py',
                 'experiments/review_async_interface.py', 'experiments/async_interface.py',
                 'experiments/prediction_error_budget.py',
                 'experiments/async_interface_model.py', 'experiments/neighbor_prediction.py'):
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    paths = sorted(p for p in args.output.rglob('*') if p.is_file())
    (args.output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(args.output)}\n' for p in paths))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
