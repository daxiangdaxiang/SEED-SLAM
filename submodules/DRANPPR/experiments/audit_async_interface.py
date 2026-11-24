"""Fixed-quadratic asynchronous communication audit; no nonlinear/ATE claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

from experiments.async_interface import Scenario, simulate
from experiments.async_interface_model import normalize, physical_factors, pose3_factors
from experiments.network_noise_support import assemble

ROOT = Path(__file__).resolve().parents[1]


def cases():
    transfer = 'results/dranppr_network_noise_transfer_qr_20261003'
    extended = 'results/dranppr_extended_calibration_20261030'
    result = {f'd{i}': f'{transfer}/d{i}/training/linearization.json' for i in (1, 4)}
    result.update({f'd{i}': f'{extended}/d{i}/training/linearization.json' for i in (2, 3, 9)})
    result.update({f'se3_{seed}': f'results/dranppr_pose3_calibration_20261117/nominal_{seed}/nominal/training.json'
                   for seed in (31, 32, 33)})
    return result


def full_reference(data, model):
    a, rhs, _ = assemble(data, sparse_output=True)
    h, b = (a.T@a).tocsc(), np.asarray(a.T@rhs)
    solution = spsolve(h, b)
    # Independently reproduce the existing global assembler's column order.
    columns, offset = {}, 0
    for factor in data['factors']:
        for term in factor['terms']:
            identity = (term['kind'], factor['robot'] if term['kind'] == 'x' else -1, term['index'])
            if identity not in columns:
                columns[identity] = offset
                offset += 3 if term['kind'] == 'x' else 2
    labels = sorted(key[2] for key in columns if key[0] == 'l')
    boundary = np.concatenate([solution[columns['l', -1, key]:columns['l', -1, key]+2] for key in labels])
    reduced = np.linalg.solve(model.h, -model.g)
    return dict(boundary_absolute_error=float(np.linalg.norm(boundary-reduced)),
        boundary_relative_error=float(np.linalg.norm(boundary-reduced)/max(np.linalg.norm(boundary), 1e-15)),
        full_relative_residual=float(np.linalg.norm(h@solution-b)/max(np.linalg.norm(b), 1e-15)))


def summarize(rows):
    complete = [r for r in rows if r['horizon_complete'] and r['finite']]
    by_key = {(r['case'], r['scenario'], r['seed'], r['mode']): r for r in complete}
    output = dict(runs=len(rows), complete_runs=len(complete),
        bound_violations=sum(r['bound_violations'] for r in rows),
        bounded_vs_stale_complete_pairs=0, bounded_vs_stale_gap_wins=0,
        bounded_vs_synchronous_complete_pairs=0, bounded_vs_synchronous_gap_wins=0)
    for row in complete:
        if row['mode'] != 'bounded':
            continue
        for other in ('stale', 'synchronous'):
            reference = by_key.get((row['case'], row['scenario'], row['seed'], other))
            if reference is not None:
                output[f'bounded_vs_{other}_complete_pairs'] += 1
                output[f'bounded_vs_{other}_gap_wins'] += int(row['relative_gap'] < reference['relative_gap']*(1.-1e-9))
    output['scope'] = 'Fixed quadratic relative objective gap; not trajectory accuracy or original-baseline performance'
    return output


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = ('experiments/async_interface_model.py', 'experiments/async_interface.py',
        'experiments/neighbor_prediction.py', 'experiments/prediction_error_budget.py',
        'experiments/audit_async_interface.py', 'experiments/network_noise_support.py',
        'experiments/native_local/pose3_calibration.cpp', 'tests/test_async_interface.py',
        'tests/test_async_interface_audit.py', 'docs/superpowers/plans/2026-11-23-asynchronous-interface.md')
    for name in source_files:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    manifest = dict(command=sys.argv, ticks=args.ticks, seeds=args.seeds, scenarios=args.scenarios,
        cases=args.cases, modes=args.modes, all_terminal=False, truth_used=False, inputs={},
        initial_state=args.initial_state,
        numpy=np.__version__, scipy=scipy.__version__,
        time_scope='Synthetic tick with uncongested routed links; not measured network/solver wall-time',
        setup_scope='Immutable pre-distributed rows and common zero increment granted to every arm; setup bytes excluded',
        comparison_scope='Matched synchronous block gradient, NOT production DRAN/PCG/iMESA/riMESA',
        reference='https://arxiv.org/html/2203.11307, Theorems 1-2, static SPD specialization')
    write_json(output/'manifest.json', manifest)
    all_rows = []
    for name in args.cases:
        folder = output/name
        folder.mkdir()
        if name == 'coupled_spd':
            model = normalize(.8*np.ones((5, 5))+.2*np.eye(5), np.ones(5), list(range(5)))
            metadata = dict(robots=list(range(5)), dimension=5, scope='Controlled SPD coupling, not SLAM data')
        else:
            source = ROOT/cases()[name]
            shutil.copy2(source, folder/'physical_input.json')
            manifest['inputs'][name] = dict(path=str(source), sha256=digest(source))
            data = json.loads(source.read_text())
            if name.startswith('se3_'):
                model, metadata = pose3_factors(data)
                if metadata['full_solve_relative_error'] > 1e-6:
                    raise ValueError('SE3 elimination disagrees with square-root reference')
            else:
                model, metadata = physical_factors(data)
                metadata['full_reference'] = full_reference(data, model)
                if metadata['full_reference']['boundary_absolute_error'] > 1e-8:
                    raise ValueError('Schur/full system mismatch')
        ids = metadata['robots']
        links = list(zip(ids[:-1], ids[1:]))
        metadata.update(condition_normalized=model.condition, owners=sorted(model.indices),
            initial_reduced_gradient_norm=float(np.linalg.norm(model.g)),
            owner_dimensions={i: len(indices) for i, indices in model.indices.items()},
            row_array_storage_bytes=sum(block.nbytes for row in model.rows().values() for block in row.values())+model.c.nbytes,
            links=links)
        write_json(folder/'model.json', metadata)
        np.savez_compressed(folder/'quadratic.npz', h=model.h, g=model.g, b=model.b, c=model.c,
                            transform=model.transform)
        # A fixed, truth-free common perturbation tests more than the tiny
        # residual left by the archived converged linearization. H and g stay fixed.
        initial = np.sin(np.arange(len(model.c))+1.) if args.initial_state == 'unit' else np.zeros_like(model.c)
        for scenario_name in args.scenarios:
            for seed in args.seeds:
                scenario = Scenario(scenario_name, seed=seed)
                for mode in args.modes:
                    started = time.perf_counter()
                    result = simulate(model, links, scenario, mode, args.ticks, initial=initial)
                    row = dict(result['summary'], case=name, service_seconds=time.perf_counter()-started)
                    run_name = f'{scenario_name}_{seed}_{mode}'
                    write_json(folder/(run_name+'.json'), row)
                    np.savez_compressed(folder/(run_name+'.npz'), states=result['states'],
                        relative_gaps=result['relative_gaps'], relative_residuals=result['relative_residuals'],
                        counters=result['counters'])
                    all_rows.append(row)
                    print(json.dumps({k: row[k] for k in ('case','scenario','seed','mode','relative_gap',
                        'relative_residual','updates','bound_violations')}), flush=True)
        write_json(output/'manifest.json', manifest)
        write_json(output/'runs.json', all_rows)
    manifest['all_terminal'] = True
    write_json(output/'manifest.json', manifest)
    write_json(output/'summary.json', summarize(all_rows))
    fields = ('case','scenario','seed','mode','ticks_completed','relative_gap','relative_residual',
              'first_gap_1e6_tick','updates','application_bytes','max_cache_age','bound_violations',
              'dropped_packets','obsolete_or_duplicate_packets','horizon_complete','service_seconds')
    with (output/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_rows)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in files))
    print(json.dumps(summarize(all_rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=list(cases())+['coupled_spd'],
                        default=['coupled_spd']+list(cases()))
    parser.add_argument('--scenarios', nargs='+', choices=['ideal','delay','heterogeneous','loss','outage'],
                        default=['ideal','delay','heterogeneous','loss','outage'])
    parser.add_argument('--modes', nargs='+', choices=['synchronous','stale','bounded','directional'],
                        default=['synchronous','stale','bounded'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[31,32,33])
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--initial-state', choices=['zero','unit'], default='unit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
