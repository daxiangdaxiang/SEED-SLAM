"""Matched-communication history prediction on fixed physical Schur models."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import Scenario, simulate
from experiments.async_interface_model import normalize, physical_factors, pose3_factors
from experiments.audit_async_interface import ROOT, cases, digest, write_json
from experiments.review_async_interface import gap_comparison


def review_pair(model, initial, candidate, reference):
    length = min(len(candidate['states']), len(reference['states']))
    if not np.array_equal(candidate['counters'][:length], reference['counters'][:length]):
        raise ValueError('prefix budget mismatch')
    optimum = cho_solve(cho_factor(model.b, lower=True), -model.c)
    start = initial-optimum
    gap0 = .5*start@model.b@start
    metrics = {}
    for label, run in (('candidate', candidate), ('reference', reference)):
        last = run['states'][-1]
        error = last-optimum
        gap = .5*error@model.b@error/gap0
        residual = np.linalg.norm(model.b@last+model.c)/np.linalg.norm(model.b@initial+model.c)
        np.testing.assert_allclose(gap, run['relative_gaps'][-1], rtol=1e-5, atol=1e-20)
        # GEMV and GEMM accumulate differently at a stationary point. Use a
        # forward-error scale, not a relative test on a numerical zero.
        neps = len(last)*np.finfo(float).eps
        roundoff = 4.*neps/(1.-neps)*(np.linalg.norm(np.abs(model.b)@np.abs(last))+
            np.linalg.norm(model.c))/np.linalg.norm(model.b@initial+model.c)
        np.testing.assert_allclose(residual, run['relative_residuals'][-1], rtol=1e-5, atol=roundoff)
        metrics[label+'_gap'], metrics[label+'_residual'] = float(gap), float(residual)
    return dict(metrics, prefix_budget_equal=True, equal_completed_horizon=len(candidate['states']) == len(reference['states']),
        gap_result=gap_comparison(metrics['candidate_gap'], metrics['reference_gap']),
        residual_result=gap_comparison(metrics['candidate_residual'], metrics['reference_residual']))


def summarize(rows, comparisons):
    summary = dict(runs=len(rows), incomplete=sum(not r['horizon_complete'] for r in rows), predictors={})
    for mode in sorted({r['prediction'] for r in comparisons}):
        items = [r for r in comparisons if r['prediction'] == mode]
        summary['predictors'][mode] = {metric: {outcome: sum(r[metric+'_result'] == outcome for r in items)
            for outcome in ('win','tie','loss')} for metric in ('gap','residual')}
        summary['predictors'][mode]['all_prefix_budgets_equal'] = all(r['prefix_budget_equal'] for r in items)
    return summary


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = ('experiments/neighbor_prediction.py', 'experiments/async_interface.py',
        'experiments/prediction_error_budget.py',
        'experiments/async_interface_model.py', 'experiments/audit_neighbor_prediction.py',
        'experiments/audit_async_interface.py', 'experiments/review_async_interface.py',
        'experiments/network_noise_support.py', 'tests/test_neighbor_prediction.py',
        'tests/test_response_prediction.py',
        'docs/superpowers/plans/2026-11-23-history-prediction.md',
        'docs/superpowers/plans/2026-11-24-response-prediction.md')
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    manifest = dict(command=sys.argv, cases=args.cases, scenarios=args.scenarios, seeds=args.seeds,
        ticks=args.ticks, history_size=args.history_size, predictions=args.predictions, solver=args.solver,
        numpy=np.__version__, scipy=scipy.__version__, all_terminal=False, inputs={}, truth_used=False,
        initial_state='sin(k+1), same normalized quadratic', topology=args.topology,
        scope='Fixed chart/physical quadratic, not nonlinear online accuracy; synthetic uncongested network time',
        setup='Common offline rows, initial state and chart; setup bytes excluded',
        theoretical_status='Earlier bounded-delay descent guarantee is NOT asserted for extrapolated gradients')
    write_json(output/'manifest.json', manifest)
    rows, comparisons = [], []
    for case in args.cases:
        folder = output/case
        folder.mkdir()
        if case == 'coupled_spd':
            model = normalize(.8*np.ones((5,5))+.2*np.eye(5), np.ones(5), list(range(5)))
            metadata = dict(robots=list(range(5)), scope='Algebraic SPD control')
        else:
            source = ROOT/cases()[case]
            shutil.copy2(source, folder/'physical_input.json')
            manifest['inputs'][case] = dict(path=str(source), sha256=digest(source))
            data = json.loads(source.read_text())
            model, metadata = pose3_factors(data) if case.startswith('se3_') else physical_factors(data)
        ids = metadata['robots']
        links = list(zip(ids[:-1], ids[1:])) if args.topology == 'chain' else [(i,j) for i in ids for j in ids if i < j]
        metadata.update(links=links, indices={i: indices.tolist() for i, indices in model.indices.items()})
        write_json(folder/'model.json', metadata)
        np.savez_compressed(folder/'quadratic.npz', b=model.b, c=model.c, h=model.h, g=model.g, transform=model.transform)
        initial = np.sin(np.arange(len(model.c))+1.)
        for scenario_name in args.scenarios:
            for seed in args.seeds:
                reference = None
                for prediction in args.predictions:
                    started = time.perf_counter()
                    result = simulate(model, links, Scenario(scenario_name, seed=seed), args.solver,
                        args.ticks, initial, prediction=prediction, history_size=args.history_size)
                    row = dict(result['summary'], case=case, service_seconds=time.perf_counter()-started)
                    if prediction == 'hold':
                        reference = result
                        if args.parity_root:
                            archived = np.load(args.parity_root/case/f'{scenario_name}_{seed}_{args.solver}.npz')
                            np.testing.assert_array_equal(result['states'], archived['states'][:args.ticks])
                            np.testing.assert_array_equal(result['counters'], archived['counters'][:args.ticks])
                            row['archive_state_parity'] = True
                    else:
                        comparisons.append(dict(review_pair(model, initial, result, reference), case=case,
                            prediction=prediction, scenario=scenario_name, seed=seed))
                        if scenario_name == 'ideal':
                            np.testing.assert_array_equal(result['states'], reference['states'])
                    run_name = f'{scenario_name}_{seed}_{prediction}'
                    write_json(folder/(run_name+'.json'), row)
                    np.savez_compressed(folder/(run_name+'.npz'), **{key:result[key] for key in
                        ('states','relative_gaps','relative_residuals','counters','gradient_error_sq')})
                    rows.append(row)
                    print(json.dumps({k:row[k] for k in ('case','scenario','seed','prediction','relative_gap',
                        'relative_residual','gradient_error_sq_sum','held_gradient_error_sq_sum','horizon_complete')}), flush=True)
        write_json(output/'runs.json', rows)
        write_json(output/'comparisons.json', comparisons)
        write_json(output/'manifest.json', manifest)
    manifest['all_terminal'] = True
    write_json(output/'manifest.json', manifest)
    write_json(output/'summary.json', summarize(rows, comparisons))
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in files))
    print(json.dumps(summarize(rows, comparisons)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=list(cases())+['coupled_spd'], default=list(cases())+['coupled_spd'])
    parser.add_argument('--scenarios', nargs='+', choices=['ideal','delay','heterogeneous','loss','outage'],
                        default=['ideal','delay','heterogeneous','loss','outage'])
    parser.add_argument('--predictions', nargs='+', choices=['hold','secant','poly1','poly2','poly3','risk3',
                        'response_linear','response_calibrated'],
                        default=['hold','secant','poly1','poly2','poly3'])
    parser.add_argument('--solver', choices=['stale','bounded','directional'], default='directional')
    parser.add_argument('--topology', choices=['chain','complete'], default='chain')
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--history-size', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[31,32,33])
    parser.add_argument('--parity-root', type=Path)
    args = parser.parse_args()
    if args.predictions[0] != 'hold' or len(set(args.predictions)) != len(args.predictions):
        parser.error('predictions must start with hold and contain no duplicates')
    run(args)


if __name__ == '__main__':
    main()
