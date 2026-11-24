"""Read-only independent metric, budget and inexact-gradient checks."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import directional_majorizer, Scenario, simulate
from experiments.async_interface_model import NormalizedModel, normalize
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_neighbor_prediction import review_pair
from experiments.review_async_interface import verify_ledger, gap_comparison


def inexact_gradient_audit(model, initial, states, record):
    before = np.vstack([initial,states[:-1]])
    increments = states-before
    true = before@model.b+model.c
    gradient_energy, error_energy = np.zeros(len(states)), np.zeros(len(states))
    error_sq = 0.
    bounds = {tuple(map(int,key.split(':'))):value for key,value in record['delay_bounds'].items()}
    periods = {int(i):p for i,p in record['period_ticks'].items()}
    for i,row in model.rows().items():
        metric = directional_majorizer(row,i,bounds)
        indices = model.indices[i]
        active = np.arange(len(states)) % periods[i] == 0
        gradient = true[:,indices]*active[:,None]
        estimated = -increments[:,indices]@metric
        error = estimated-gradient
        factor = cho_factor(metric,lower=True)
        gradient_energy += np.einsum('ij,ij->i',gradient,cho_solve(factor,gradient.T).T)
        error_energy += np.einsum('ij,ij->i',error,cho_solve(factor,error.T).T)
        error_sq += np.sum(error**2)
    actual_change = np.einsum('ij,ij->i',true,increments)+.5*np.einsum('ij,ij->i',increments@model.b,increments)
    upper = -.5*gradient_energy+.5*error_energy
    optimum = cho_solve(cho_factor(model.b,lower=True),-model.c)
    initial_error = initial-optimum
    scale = .5*initial_error@model.b@initial_error
    return dict(max_normalized_bound_violation=float(max(0.,np.max(actual_change-upper))/scale),
        reconstructed_error_sq_sum=float(error_sq),
        sufficient_descent_steps=int(np.sum(error_energy < gradient_energy)),
        measured_decrease_steps=int(np.sum(actual_change < 0.)), total_steps=len(states))


def review(root, prior):
    manifest = verify_ledger(root)
    if manifest['solver'] != 'directional':
        raise ValueError('this audit is for the fixed directional update')
    records = json.loads((root/'runs.json').read_text())
    expected = len(manifest['cases'])*len(manifest['scenarios'])*len(manifest['seeds'])*len(manifest['predictions'])
    if len(records) != expected:
        raise ValueError('missing run')
    lookup = {(r['case'],r['scenario'],r['seed'],r['prediction']):r for r in records}
    if len(lookup) != expected:
        raise ValueError('duplicate run')
    output = dict(runs=expected, metric_checks=0, prior_state_parity=0, prefix_budget_checks=0,
        max_normalized_bound_violation=0., comparisons={}, cases={}, failures=[], comparisons_detail=[],
        heldout_cases=['d2','d3','d9','se3_32','se3_33'])
    for case in manifest['cases']:
        folder = root/case
        data = np.load(folder/'quadratic.npz')
        metadata = json.loads((folder/'model.json').read_text())
        indices = {int(i):np.asarray(ii) for i,ii in metadata['indices'].items()}
        model = NormalizedModel(data['h'],data['g'],data['b'],data['c'],data['transform'],indices,float(np.linalg.cond(data['b'])))
        initial = np.sin(np.arange(len(model.c))+1.)
        if prior:
            old = np.load(prior/case/'quadratic.npz')
            for key in ('h','g','b','c','transform'):
                np.testing.assert_array_equal(data[key],old[key])
        for scenario in manifest['scenarios']:
            for seed in manifest['seeds']:
                traces = {mode:np.load(folder/f'{scenario}_{seed}_{mode}.npz') for mode in manifest['predictions']}
                reference = traces['hold']
                if prior:
                    old = np.load(prior/case/f'{scenario}_{seed}_directional.npz')
                    np.testing.assert_array_equal(old['states'][:manifest['ticks']],reference['states'])
                    output['prior_state_parity'] += 1
                for mode,trace in traces.items():
                    record = lookup[case,scenario,seed,mode]
                    if not record['horizon_complete'] or len(trace['states']) != manifest['ticks']:
                        raise ValueError('incomplete run: '+str((case,scenario,seed,mode)))
                    metrics = review_pair(model,initial,trace,reference)
                    output['metric_checks'] += 1
                    if mode != 'hold':
                        output['prefix_budget_checks'] += 1
                    if scenario == 'ideal':
                        np.testing.assert_array_equal(trace['states'],reference['states'])
                    audit = inexact_gradient_audit(model,initial,trace['states'],record)
                    if audit['max_normalized_bound_violation'] > 1e-10:
                        raise ValueError('inexact-gradient algebra check failed')
                    np.testing.assert_allclose(audit['reconstructed_error_sq_sum'],trace['gradient_error_sq'][:,0].sum(),rtol=1e-7,atol=1e-20)
                    output['max_normalized_bound_violation'] = max(output['max_normalized_bound_violation'],audit['max_normalized_bound_violation'])
                    if mode == 'hold':
                        continue
                    item = dict(metrics,case=case,scenario=scenario,seed=seed,prediction=mode,
                        relative_gap_reduction=1.-metrics['candidate_gap']/max(metrics['reference_gap'],1e-30),
                        conditional_descent_fraction=audit['sufficient_descent_steps']/len(trace['states']),
                        prediction_error_ratio=record['gradient_error_sq_sum']/max(record['held_gradient_error_sq_sum'],1e-30))
                    output['comparisons_detail'].append(item)
                    for other in ('hold','secant','poly2'):
                        if mode == other or other not in traces:
                            continue
                        key = mode+'_vs_'+other
                        result = gap_comparison(trace['relative_gaps'][-1],traces[other]['relative_gaps'][-1])
                        counts = output['comparisons'].setdefault(key,dict(win=0,tie=0,loss=0))
                        counts[result] += 1
                    if mode == 'risk3' and item['gap_result'] == 'loss':
                        output['failures'].append(item)
        for mode in manifest['predictions']:
            items = [r for r in output['comparisons_detail'] if r['case']==case and r['prediction']==mode and r['scenario']!='ideal']
            if items:
                output['cases'][case+':'+mode] = dict(gap={outcome:sum(r['gap_result']==outcome for r in items) for outcome in ('win','tie','loss')},
                    median_nonideal_gap_reduction=float(np.median([r['relative_gap_reduction'] for r in items])),
                    median_prediction_error_ratio=float(np.median([r['prediction_error_ratio'] for r in items])))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result = review(args.root,args.prior)
    counter = normalize(.8*np.ones((5,5))+.2*np.eye(5),np.ones(5),list(range(5)))
    links = [(i,j) for i in range(5) for j in range(i+1,5)]
    result['counterexample'] = []
    for solver in ('stale','directional'):
        for prediction in ('hold','secant','poly2','risk3'):
            run = simulate(counter,links,Scenario('delay',4,4),solver,1000,
                np.sin(np.arange(5)+1.),prediction=prediction)
            result['counterexample'].append(run['summary'])
            np.savez_compressed(args.output/f'counter_{solver}_{prediction}.npz',**{
                key:run[key] for key in ('states','relative_gaps','counters','gradient_error_sq')})
    write_json(args.output/'review.json',result)
    write_json(args.output/'manifest.json',dict(all_terminal=True,root=str(args.root),
        source_manifest_sha256=digest(args.root/'manifest.json'),scope='Independent artifact review; no optimization decisions'))
    for name in ('experiments/review_history_prediction.py','experiments/audit_neighbor_prediction.py',
        'experiments/review_async_interface.py','experiments/async_interface.py',
        'experiments/prediction_error_budget.py',
        'experiments/neighbor_prediction.py','experiments/async_interface_model.py','tests/test_neighbor_prediction.py'):
        target=args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    files=sorted(p for p in args.output.rglob('*') if p.is_file())
    (args.output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(args.output)}\n' for p in files))
    print(json.dumps({key:value for key,value in result.items() if key not in ('comparisons_detail','failures','counterexample')},indent=2))


if __name__ == '__main__':
    main()
