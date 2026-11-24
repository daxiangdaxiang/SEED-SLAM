"""Independent sealed-trace review; forward-error-aware coverage is offline only."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import directional_majorizer
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_error_budget import audit_updates
from experiments.audit_neighbor_prediction import review_pair
from experiments.audit_response_controls import load_model, seal
from experiments.review_async_interface import verify_ledger


def error_coverage(model, initial, run):
    before = np.vstack([initial, run['states'][:-1]])
    actual = before@model.b+model.c
    # Scale with the operands, not the nearly cancelling resulting gradient.
    # This corrects only offline floating-point diagnostics, never actor bounds.
    neps = len(model.c)*np.finfo(float).eps
    forward = 8.*neps/(1.-neps)*(np.abs(before)@np.abs(model.b)+np.abs(model.c))
    zero = {(i, j): 0 for i in model.indices for j in model.indices}
    counts = dict(supported=0, covered=0, moving=0, moving_covered=0, zero=0)
    windows = [dict(start=int(part[0]), end=int(part[-1])+1, moving=0, moving_covered=0)
               for part in np.array_split(np.arange(len(before)), min(4, len(before)))]
    for rank, (i, row) in enumerate(sorted(model.rows().items())):
        ii = model.indices[i]
        metric = directional_majorizer(row, i, zero)
        p0 = cho_solve(cho_factor(metric, lower=True), np.eye(len(metric)))
        trace = run['error_budget_trace'][:, rank]
        ready, epsilon, alpha = trace[:, 0] > 0., trace[:, 1], trace[:, 2]
        error = run['estimated_gradients'][:, ii]-actual[:, ii]
        norm = np.sqrt(np.maximum(0., np.einsum('ij,ij->i', error@p0, error)))
        tolerance = np.sqrt(np.maximum(0., np.einsum('ij,ij->i',
            forward[:, ii]@np.abs(p0), forward[:, ii])))
        within = norm <= epsilon+tolerance
        moving = ready & (alpha > 0.)
        counts['supported'] += int(ready.sum())
        counts['covered'] += int((ready & within).sum())
        counts['moving'] += int(moving.sum())
        counts['moving_covered'] += int((moving & within).sum())
        counts['zero'] += int((ready & ~moving).sum())
        for window in windows:
            take = slice(window['start'], window['end'])
            window['moving'] += int(moving[take].sum())
            window['moving_covered'] += int((moving & within)[take].sum())
    counts['windows'] = windows
    return counts


def aggregate(comparisons):
    result = {}
    for label in sorted({r['label'] for r in comparisons}):
        selected = [r for r in comparisons if r['label'] == label]
        groups = dict(all=selected, nonideal=[r for r in selected if r['scenario'] != 'ideal'])
        groups.update({case: [r for r in selected if r['case'] == case and r['scenario'] != 'ideal']
                       for case in sorted({r['case'] for r in selected})})
        for scope, rows in groups.items():
            if rows:
                result[label+':'+scope] = dict(
                    counts={key: sum(r['gap_result'] == key for r in rows) for key in ('win', 'tie', 'loss')},
                    median_gap_reduction=float(np.median([r['gap_reduction'] for r in rows])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path)
    args = parser.parse_args()
    manifest = verify_ledger(args.root)
    if args.history:
        verify_ledger(args.history)
        if manifest.get('stress') or manifest.get('initial_seed') is not None:
            parser.error('history comparison requires the original common protocol')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.root/'runs.json').read_text())
    lookup = {(r['case'], r['scenario'], r['seed'], r['arm']): r for r in rows}
    expected = len(manifest['cases'])*len(manifest['scenarios'])*len(manifest['seeds'])*len(manifest['arms'])
    assert len(lookup) == len(rows) == expected
    comparisons, coverage, hits = [], [], []
    worst_update = 0.
    for case in manifest['cases']:
        model, _ = load_model(args.root, case)
        initial_file = args.root/case/'initial.npy'
        initial = np.load(initial_file) if initial_file.exists() else np.sin(np.arange(len(model.c))+1.)
        for net in manifest['scenarios']:
            for seed in manifest['seeds']:
                runs = {arm: dict(np.load(args.root/case/f'{net}_{seed}_{arm}.npz')) for arm in manifest['arms']}
                for arm, run in runs.items():
                    run['summary'] = lookup[case, net, seed, arm]
                    checked = audit_updates(model, run, initial)
                    worst_update = max(worst_update, checked['update_discrepancy'])
                    review_pair(model, initial, run, runs['baseline'])
                    if net == 'ideal':
                        np.testing.assert_array_equal(run['states'], runs['baseline']['states'])
                    if arm.startswith('budget_'):
                        coverage.append(dict(case=case, scenario=net, seed=seed, arm=arm,
                                             **error_coverage(model, initial, run)))
                if args.history:
                    for name in ('hold', 'risk3'):
                        runs[name] = dict(np.load(args.history/case/f'{net}_{seed}_{name}.npz'))
                    np.testing.assert_array_equal(runs['hold']['states'], runs['baseline']['states'])
                pairs = [('prediction_only', 'baseline'), ('budget_hold', 'baseline'),
                         ('budget_predict', 'baseline'), ('budget_predict', 'prediction_only'),
                         ('budget_predict', 'budget_hold'), ('unchecked_predict', 'baseline')]
                if args.history:
                    pairs.append(('budget_predict', 'risk3'))
                for arm, other in pairs:
                    if arm not in runs or other not in runs:
                        continue
                    a, b = runs[arm], runs[other]
                    checked = review_pair(model, initial, a, b)
                    comparisons.append(dict(case=case, scenario=net, seed=seed, label=arm+'_vs_'+other,
                        **checked, gap_reduction=1.-checked['candidate_gap']/max(checked['reference_gap'], 1e-30)))
                for target_name, target in [('fixed_1e6', 1e-6),
                        ('baseline_endpoint', float(runs['baseline']['relative_gaps'][-1])*(1.+1e-12))]:
                    for arm, run in runs.items():
                        crossing = np.flatnonzero(run['relative_gaps'] <= target)
                        tick = int(crossing[0]) if len(crossing) else None
                        hits.append(dict(case=case, scenario=net, seed=seed, arm=arm,
                            target_name=target_name, target=target, ticks=None if tick is None else tick+1,
                            bytes=None if tick is None else int(run['counters'][tick, 0])))
    totals = {}
    for arm in ('budget_hold', 'budget_predict'):
        for scope in ('all', 'nonideal', 'ideal'):
            selected = [r for r in coverage if r['arm'] == arm and
                        (scope == 'all' or (r['scenario'] == 'ideal') == (scope == 'ideal'))]
            counts = {key: sum(r[key] for r in selected)
                      for key in ('supported', 'covered', 'moving', 'moving_covered', 'zero')}
            counts['coverage'] = counts['covered']/max(1, counts['supported'])
            counts['moving_coverage'] = counts['moving_covered']/max(1, counts['moving'])
            totals[arm+':'+scope] = counts
    summary = dict(runs=expected, incomplete=sum(not r['horizon_complete'] for r in rows),
        comparisons=aggregate(comparisons), coverage=totals,
        all_prefix_budgets_equal=all(r['prefix_budget_equal'] for r in comparisons),
        max_normalized_update_discrepancy=worst_update)
    for name, value in [('summary', summary), ('comparisons', comparisons), ('coverage', coverage), ('target_hits', hits)]:
        write_json(args.output/(name+'.json'), value)
    for path in sorted((ROOT/'experiments').glob('*.py')):
        if path.name in ('review_error_budget.py', 'audit_error_budget.py', 'async_interface.py',
                'prediction_error_budget.py', 'neighbor_prediction.py', 'audit_async_interface.py',
                'audit_neighbor_prediction.py', 'async_interface_model.py', 'audit_response_controls.py',
                'review_async_interface.py', 'audit_history_forecast.py', 'network_noise_support.py'):
            target = args.output/'source'/'experiments'/path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    seal(args.output, dict(root=str(args.root.resolve()), root_ledger=digest(args.root/'SHA256SUMS'),
        history=None if not args.history else str(args.history.resolve()),
        history_ledger=None if not args.history else digest(args.history/'SHA256SUMS'),
        coverage='Operand-scaled floating point tolerance; also report moving steps separately',
        scope='Offline audit only; no bounds or decisions are changed'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
