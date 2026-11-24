"""Same-horizon empirical error envelopes coupled to local update strength."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import Scenario, simulate, directional_majorizer
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_neighbor_prediction import review_pair
from experiments.audit_response_controls import load_model, seal
from experiments.review_async_interface import verify_ledger, gap_comparison


ARMS = dict(baseline=('directional', 'hold'), prediction_only=('directional', 'response_linear'),
    budget_hold=('error_budget', 'hold'), budget_predict=('error_budget', 'response_linear'),
    unchecked_predict=('fresh_unchecked', 'response_linear'))


def audit_updates(model, run, initial):
    states, record = run['states'], run['summary']
    before = np.vstack([initial, states[:-1]])
    delta = states-before
    actual = before@model.b+model.c
    neps = len(model.c)*np.finfo(float).eps
    forward_error = 8.*neps/(1.-neps)*(np.abs(before)@np.abs(model.b)+np.abs(model.c))
    estimated = run['estimated_gradients']
    bounds = {tuple(map(int, key.split(':'))): value for key, value in record['delay_bounds'].items()}
    zero = {key: 0 for key in bounds}
    periods = {int(i): value for i, value in record['period_ticks'].items()}
    supported = covered = 0
    all_supported, all_covered = np.ones(len(states), bool), np.ones(len(states), bool)
    descent_bound = np.zeros(len(states))
    step_discrepancy = 0.
    for rank, (i, row) in enumerate(sorted(model.rows().items())):
        ii = model.indices[i]
        ghat = estimated[:, ii]
        active = np.arange(len(states)) % periods[i] == 0
        fresh = directional_majorizer(row, i, zero)
        p0 = cho_solve(cho_factor(fresh, lower=True), np.eye(len(fresh)))
        delayed = directional_majorizer(row, i, bounds)
        pd = cho_solve(cho_factor(delayed, lower=True), np.eye(len(delayed)))
        expected = -ghat@(p0 if record['mode'] == 'fresh_unchecked' else pd)
        if record['mode'] == 'error_budget':
            trace = run['error_budget_trace'][:, rank]
            ready, epsilon, alpha = trace[:, 0] > 0., trace[:, 1], trace[:, 2]
            assert not np.any(ready & ~active)
            assert np.all(epsilon[ready] >= 0.)
            gnorm = np.sqrt(np.maximum(0., np.einsum('ij,ij->i', ghat@p0, ghat)))
            reconstructed_alpha = np.maximum(0., 1.-np.divide(epsilon, gnorm,
                out=np.ones_like(epsilon), where=gnorm > 0.))
            np.testing.assert_allclose(alpha[ready], reconstructed_alpha[ready], rtol=1e-9, atol=1e-12)
            expected[ready] = -alpha[ready, None]*(ghat[ready]@p0)
            error = ghat-actual[:, ii]
            error_norm = np.sqrt(np.maximum(0., np.einsum('ij,ij->i', error@p0, error)))
            tolerance = np.sqrt(np.maximum(0., np.einsum('ij,ij->i',
                forward_error[:, ii]@np.abs(p0), forward_error[:, ii])))
            within = error_norm <= epsilon+tolerance
            supported += int(ready.sum())
            covered += int((ready & within).sum())
            all_supported &= ready | ~active
            all_covered &= (ready & within) | ~active
            descent_bound -= .5*np.square(alpha*gnorm)
        local_scale = np.maximum(1., np.maximum(np.abs(before[:, ii]), np.abs(states[:, ii])))
        discrepancy = float(np.max(np.abs(delta[:, ii]-expected)/local_scale))
        step_discrepancy = max(step_discrepancy, discrepancy)
        if discrepancy > 1e-12:
            raise ValueError('stored update does not match its declared local policy')
    change = np.einsum('ij,ij->i', actual, delta)+.5*np.einsum('ij,ij->i', delta@model.b, delta)
    certified = all_supported & all_covered if record['mode'] == 'error_budget' else np.zeros(len(states), bool)
    scale = max(1., float(initial@model.b@initial))
    violation = float(max(0., np.max((change-descent_bound)[certified])/scale)) if certified.any() else 0.
    if violation > 1e-12:
        raise ValueError('conditional fresh-majorizer descent identity failed')
    return dict(update_discrepancy=step_discrepancy, supported_updates=supported,
        actual_error_covered=covered, actual_error_coverage=covered/supported if supported else None,
        fully_supported_ticks=int(all_supported.sum()) if record['mode'] == 'error_budget' else 0,
        certified_ticks=int(certified.sum()), conditional_bound_violation=violation,
        objective_decrease_ticks=int((change <= 0.).sum()))


def summarize(rows):
    lookup = {(r['case'], r['scenario'], r['seed'], r['arm']): r for r in rows}
    output = dict(runs=len(rows), incomplete=sum(not r['horizon_complete'] for r in rows), comparisons={})
    for arm, other in [('prediction_only', 'baseline'), ('budget_hold', 'baseline'),
                       ('budget_predict', 'baseline'), ('budget_predict', 'prediction_only'),
                       ('budget_predict', 'budget_hold'), ('unchecked_predict', 'baseline')]:
        pairs = [(r, lookup[r['case'], r['scenario'], r['seed'], other]) for r in rows
                 if r['arm'] == arm and (r['case'], r['scenario'], r['seed'], other) in lookup]
        if not pairs:
            continue
        counts = {key: 0 for key in ('win', 'tie', 'loss')}
        for a, b in pairs:
            counts[gap_comparison(a['relative_gap'], b['relative_gap'])] += 1
        output['comparisons'][arm+'_vs_'+other] = counts
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/dranppr_response_final_20261124')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--scenarios', nargs='+', default=['ideal', 'delay', 'heterogeneous', 'loss', 'outage'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[31, 32, 33])
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--memory', type=int, default=8)
    parser.add_argument('--initial-seed', type=int)
    args = parser.parse_args()
    previous = verify_ledger(args.root)
    cases = args.cases or previous['cases']
    if args.arms[0] != 'baseline':
        parser.error('first arm must be baseline')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ('experiments/audit_error_budget.py', 'experiments/async_interface.py',
        'experiments/prediction_error_budget.py', 'experiments/neighbor_prediction.py',
        'experiments/async_interface_model.py', 'experiments/audit_async_interface.py',
        'experiments/audit_neighbor_prediction.py', 'experiments/audit_response_controls.py',
        'experiments/audit_history_forecast.py', 'experiments/review_async_interface.py',
        'experiments/network_noise_support.py', 'tests/test_prediction_error_budget.py',
        'docs/superpowers/plans/2026-11-24-horizon-error-budget.md')
    for name in sources:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    manifest = dict(command=sys.argv, all_terminal=False, cases=cases, scenarios=args.scenarios,
        seeds=args.seeds, ticks=args.ticks, arms=args.arms, source_root=str(args.root.resolve()),
        source_ledger=digest(args.root/'SHA256SUMS'), memory=args.memory, stress=args.stress,
        initial_seed=args.initial_seed,
        scope='Fixed quadratics; empirical errors not certified future bounds; no native/ATE claim',
        setup_bytes_included=False, truth_used=False, bound_definition='Exact-age last-L maximum')
    write_json(args.output/'manifest.json', manifest)
    rows = []
    for case in cases:
        folder = args.output/case
        folder.mkdir()
        model, links = load_model(args.root, case)
        for name in ('quadratic.npz', 'model.json'):
            shutil.copy2(args.root/case/name, folder/name)
        if args.stress and case == 'coupled_spd':
            links = [(i, j) for i in range(5) for j in range(i+1, 5)]
            metadata = json.loads((folder/'model.json').read_text())
            metadata['links'] = links
            write_json(folder/'model.json', metadata)
        initial = np.sin(np.arange(len(model.c))+1.)
        if args.initial_seed is not None:
            initial = np.random.default_rng(args.initial_seed).normal(size=len(model.c))
        np.save(folder/'initial.npy', initial)
        for net in args.scenarios:
            for seed in args.seeds:
                scenario = (Scenario(net, 12, 20, seed=seed) if net == 'delay' else
                    Scenario(net, seed=seed, outage=(200, 600))) if args.stress else Scenario(net, seed=seed)
                reference = None
                for arm in args.arms:
                    mode, prediction = ARMS[arm]
                    started = time.perf_counter()
                    result = simulate(model, links, scenario, mode, args.ticks, initial,
                        prediction=prediction, history_size=args.memory)
                    elapsed = time.perf_counter()-started
                    if reference is None:
                        reference = result
                    comparison = review_pair(model, initial, result, reference)
                    parity = False
                    if (mode == 'directional' and not args.stress and args.initial_seed is None and
                            (prediction == 'hold' or args.memory == 8)):
                        old = np.load(args.root/case/f'{net}_{seed}_{prediction}.npz')
                        prefix = min(args.ticks, len(old['states']))
                        np.testing.assert_array_equal(result['states'][:prefix], old['states'][:prefix])
                        parity = True
                    if net == 'ideal':
                        np.testing.assert_array_equal(result['states'], reference['states'])
                    checked = audit_updates(model, result, initial)
                    row = dict(result['summary'], arm=arm, case=case, service_seconds=elapsed,
                        comparison=comparison, audit=checked, archive_state_parity=parity)
                    rows.append(row)
                    name = f'{net}_{seed}_{arm}'
                    write_json(folder/(name+'.json'), row)
                    np.savez_compressed(folder/(name+'.npz'), **{key: result[key] for key in
                        ('states', 'relative_gaps', 'relative_residuals', 'counters', 'gradient_error_sq',
                         'estimated_gradients', 'error_budget_trace')})
                    print(json.dumps(dict(case=case, network=net, seed=seed, arm=arm,
                        gap=row['relative_gap'], complete=row['horizon_complete'],
                        supported=checked['supported_updates'], coverage=checked['actual_error_coverage'])), flush=True)
        write_json(args.output/'runs.json', rows)
        write_json(args.output/'summary.json', summarize(rows))
    seal(args.output, manifest)
    print(json.dumps(summarize(rows)), flush=True)


if __name__ == '__main__':
    main()
