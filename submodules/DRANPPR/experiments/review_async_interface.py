"""Independent artifact checks and a fixed delayed-SPD counterexample."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import Scenario, simulate, directional_majorizer
from experiments.async_interface_model import normalize, physical_factors, pose3_factors
from experiments.audit_async_interface import ROOT, digest, write_json


def gap_comparison(candidate, reference):
    tolerance = max(1e-20, 1e-8*max(candidate, reference))
    if abs(candidate-reference) <= tolerance:
        return 'tie'
    return 'win' if candidate < reference else 'loss'


def coupling_radius(matrix, dimensions):
    offsets = np.r_[0, np.cumsum(dimensions)]
    comparison = np.zeros((len(dimensions), len(dimensions)))
    for i in range(len(dimensions)):
        for j in range(len(dimensions)):
            if i != j:
                comparison[i,j] = np.linalg.norm(matrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]], 2)
    return float(np.max(np.abs(np.linalg.eigvals(comparison))))


def lyapunov_audit(model, states, initial, serialized_bounds):
    """Offline check of the fixed-model derivation; never an acceptance gate."""
    bounds = {tuple(map(int,key.split(':'))): value for key,value in serialized_bounds.items()}
    sequence = np.vstack([initial,states])
    increments = np.diff(sequence,axis=0)
    optimum = cho_solve(cho_factor(model.b,lower=True),-model.c)
    error = sequence-optimum
    energy = .5*np.einsum('ij,ij->i',error@model.b,error)
    step_energy = np.zeros(len(states))
    rows = model.rows()
    for i, row in rows.items():
        delta_i = increments[:,model.indices[i]]
        metric = directional_majorizer(row,i,bounds)
        step_energy += np.einsum('ij,ij->i',delta_i@metric,delta_i)
        for j, block in row.items():
            if j == i or bounds[i,j] == 0:
                continue
            _, singular, vt = np.linalg.svd(block,full_matrices=False)
            right_curvature = (vt.T*singular)@vt
            delta_j = increments[:,model.indices[j]]
            terms = np.einsum('ij,ij->i',delta_j@right_curvature,delta_j)
            delay = bounds[i,j]
            energy[1:] += .5*np.convolve(terms,np.arange(delay,0,-1),mode='full')[:len(states)]
    violations = np.diff(energy)+.5*step_energy
    return dict(max_normalized_violation=float(max(0.,np.max(violations))/max(energy[0],1e-30)),
                initial_energy=float(energy[0]),final_energy=float(energy[-1]))


def verify_ledger(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if digest(root/name) != expected:
            raise ValueError('archive changed: '+str(root/name))
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest['all_terminal']:
        raise ValueError('unfinished experiment')
    return manifest


def verify_archive(root):
    manifest = verify_ledger(root)
    rows = json.loads((root/'runs.json').read_text())
    expected = len(manifest['cases'])*len(manifest['scenarios'])*len(manifest['seeds'])*len(manifest['modes'])
    if len(rows) != expected or len({(r['case'],r['scenario'],r['seed'],r['mode']) for r in rows}) != expected:
        raise ValueError('missing or repeated run')
    checks = dict(root=str(root), runs=len(rows), ideal_state_parity_pairs=0,
        equal_prefix_byte_pairs=0, independent_metric_checks=0, max_relative_metric_error=0.,
        bound_violations=sum(r['bound_violations'] for r in rows),
        incomplete_runs=sum(not r['horizon_complete'] for r in rows), models={}, comparisons={}, tables={})
    for case in manifest['cases']:
        folder = root/case
        data = np.load(folder/'quadratic.npz')
        b, c = data['b'], data['c']
        if case == 'coupled_spd':
            dimensions, order = [1]*5, np.arange(5)
        else:
            physical = json.loads((folder/'physical_input.json').read_text())
            model, _ = pose3_factors(physical) if case.startswith('se3_') else physical_factors(physical)
            order = np.concatenate(list(model.indices.values()))
            dimensions = [len(ii) for ii in model.indices.values()]
        checks['models'][case] = dict(comparison_spectral_radius=coupling_radius(b[np.ix_(order,order)], dimensions),
            normalized_condition=float(np.linalg.cond(b)))
        optimum = cho_solve(cho_factor(b, lower=True), -c)
        initial = np.sin(np.arange(len(c))+1.) if manifest.get('initial_state','zero') == 'unit' else np.zeros_like(c)
        initial_error = initial-optimum
        initial_gap = .5*initial_error@b@initial_error
        for scenario in manifest['scenarios']:
            for seed in manifest['seeds']:
                traces = {mode: np.load(folder/f'{scenario}_{seed}_{mode}.npz') for mode in manifest['modes']}
                for mode, trace in traces.items():
                    error = trace['states'][-1]-optimum
                    gap = .5*error@b@error/initial_gap
                    discrepancy = abs(gap-trace['relative_gaps'][-1])/max(1e-20, gap, trace['relative_gaps'][-1])
                    if discrepancy > 1e-5:
                        raise ValueError('independent metric mismatch')
                    checks['max_relative_metric_error'] = max(checks['max_relative_metric_error'], float(discrepancy))
                    checks['independent_metric_checks'] += 1
                    if mode == 'synchronous':
                        continue
                    reference = traces['synchronous']
                    if not np.array_equal(trace['counters'][:,0], reference['counters'][:,0]):
                        raise ValueError('prefix-byte mismatch')
                    checks['equal_prefix_byte_pairs'] += 1
                    if scenario == 'ideal':
                        if not np.array_equal(trace['states'], reference['states']):
                            raise ValueError('zero-delay state mismatch')
                        checks['ideal_state_parity_pairs'] += 1
                    if scenario != 'ideal':
                        label = mode+'_vs_synchronous'
                        counts = checks['comparisons'].setdefault(label, dict(win=0,tie=0,loss=0))
                        counts[gap_comparison(trace['relative_gaps'][-1], reference['relative_gaps'][-1])] += 1
                        budget = reference['counters'][-1,1]
                        eligible = np.flatnonzero(trace['counters'][:,1] <= budget)
                        if len(eligible):
                            counts = checks['comparisons'].setdefault(label+'_equal_updates', dict(win=0,tie=0,loss=0))
                            counts[gap_comparison(trace['relative_gaps'][eligible[-1]], reference['relative_gaps'][-1])] += 1
            checks['tables'][case+':'+scenario] = {
                mode: dict(median_gap=float(np.median([r['relative_gap'] for r in rows
                    if r['case']==case and r['scenario']==scenario and r['mode']==mode])),
                    median_residual=float(np.median([r['relative_residual'] for r in rows
                    if r['case']==case and r['scenario']==scenario and r['mode']==mode])))
                for mode in manifest['modes']}
    return checks


def compare_candidate(reference, candidate):
    old_manifest, manifest = verify_ledger(reference), verify_ledger(candidate)
    for field in ('cases','scenarios','seeds','ticks','initial_state','inputs'):
        if manifest[field] != old_manifest[field]:
            raise ValueError('candidate protocol mismatch: '+field)
    records = json.loads((candidate/'runs.json').read_text())
    references = {(r['case'],r['scenario'],r['seed'],r['mode']):r
                  for r in json.loads((reference/'runs.json').read_text())}
    expected = len(manifest['cases'])*len(manifest['scenarios'])*len(manifest['seeds'])
    if len(records) != expected or any(r['mode'] != 'directional' for r in records):
        raise ValueError('incomplete directional matrix')
    output = dict(runs=len(records),prefix_bytes_and_updates_equal_pairs=0,
        max_normalized_lyapunov_violation=0., bound_violations=sum(r['bound_violations'] for r in records),
        comparisons={name:dict(win=0,tie=0,loss=0) for name in ('bounded','synchronous','stale')},
        cases={},rows=[])
    for case in manifest['cases']:
        folder = candidate/case
        candidate_model, baseline_model = np.load(folder/'quadratic.npz'), np.load(reference/case/'quadratic.npz')
        for key in ('h','g','b','c','transform'):
            if not np.array_equal(candidate_model[key],baseline_model[key]):
                raise ValueError('candidate changed physical model')
        if case == 'coupled_spd':
            model=normalize(candidate_model['h'],candidate_model['g'],list(range(5)))
        else:
            physical=json.loads((folder/'physical_input.json').read_text())
            model,_=pose3_factors(physical) if case.startswith('se3_') else physical_factors(physical)
        initial=np.sin(np.arange(len(model.c))+1.)
        local_records=[r for r in records if r['case']==case]
        case_counts=dict(win=0,tie=0,loss=0)
        reductions=[]
        for row in local_records:
            name=f"{row['scenario']}_{row['seed']}_"
            trace=np.load(folder/(name+'directional.npz'))
            previous=np.load(reference/case/(name+'bounded.npz'))
            if not row['horizon_complete'] or not row['finite']:
                raise ValueError('directional run did not finish')
            if not np.array_equal(trace['counters'],previous['counters']):
                raise ValueError('different bytes or update opportunities')
            output['prefix_bytes_and_updates_equal_pairs'] += 1
            audit=lyapunov_audit(model,trace['states'],initial,row['delay_bounds'])
            output['max_normalized_lyapunov_violation']=max(output['max_normalized_lyapunov_violation'],audit['max_normalized_violation'])
            if audit['max_normalized_violation'] > 1e-10:
                raise ValueError('directional Lyapunov inequality failed')
            comparisons={}
            for mode in output['comparisons']:
                old=references[case,row['scenario'],row['seed'],mode]
                classification=gap_comparison(row['relative_gap'],old['relative_gap'])
                output['comparisons'][mode][classification] += 1
                comparisons[mode]=classification
                if mode=='bounded':
                    case_counts[classification] += 1
                    if old['relative_gap'] > 1e-20:
                        reductions.append(100*(1.-row['relative_gap']/old['relative_gap']))
            output['rows'].append(dict(case=case,scenario=row['scenario'],seed=row['seed'],
                relative_gap=row['relative_gap'],comparison=comparisons,lyapunov=audit))
        output['cases'][case]=dict(comparison_to_bounded=case_counts,
            median_gap_reduction_percent=float(np.median(reductions)) if reductions else None)
    return output


def counterexample(output):
    model = normalize(.8*np.ones((5,5))+.2*np.eye(5), np.ones(5), list(range(5)))
    links = [(i,j) for i in range(5) for j in range(i+1,5)]
    initial = np.sin(np.arange(5)+1.)
    result = dict(scope='Analytic SPD stress control, complete communication graph, uniform four-tick delay',
                  min_eigenvalue=float(np.linalg.eigvalsh(model.h)[0]), runs={})
    for mode in ('synchronous','stale','bounded','directional'):
        run = simulate(model, links, Scenario('delay',4,4), mode, 1000, initial=initial)
        result['runs'][mode] = run['summary']
        np.savez_compressed(output/f'counterexample_{mode}.npz', states=run['states'],
                            gaps=run['relative_gaps'], counters=run['counters'])
    if result['runs']['stale']['relative_gap'] < 1e20 or result['runs']['bounded']['relative_gap'] > 1e-3:
        raise ValueError('counterexample no longer reproduced')
    write_json(output/'counterexample.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('experiments/review_async_interface.py','experiments/async_interface.py',
                 'experiments/prediction_error_budget.py','experiments/neighbor_prediction.py',
                 'experiments/async_interface_model.py','tests/test_review_async_interface.py',
                 'tests/test_async_interface.py'):
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    reviews = [verify_archive(root.resolve()) for root in args.roots]
    write_json(args.output/'reviews.json', reviews)
    candidate = None
    if args.candidate_root:
        candidate=compare_candidate(args.roots[-1].resolve(),args.candidate_root.resolve())
        write_json(args.output/'candidate.json',candidate)
    control = counterexample(args.output)
    result = dict(archives=len(reviews), runs=sum(r['runs'] for r in reviews),
        independent_metric_checks=sum(r['independent_metric_checks'] for r in reviews),
        ideal_state_parity_pairs=sum(r['ideal_state_parity_pairs'] for r in reviews),
        equal_prefix_byte_pairs=sum(r['equal_prefix_byte_pairs'] for r in reviews),
        bound_violations=sum(r['bound_violations'] for r in reviews),
        incomplete_runs=sum(r['incomplete_runs'] for r in reviews),
        counterexample={mode: values['relative_gap'] for mode,values in control['runs'].items()})
    if candidate:
        result['candidate']={key:candidate[key] for key in ('runs','prefix_bytes_and_updates_equal_pairs',
            'max_normalized_lyapunov_violation','bound_violations','comparisons','cases')}
    write_json(args.output/'summary.json', result)
    files = sorted(p for p in args.output.rglob('*') if p.is_file())
    (args.output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(args.output)}\n' for p in files))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
