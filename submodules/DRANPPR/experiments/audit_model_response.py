"""Sealed model-response experiments with incremental payloads fully charged."""
import argparse
import heapq
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from experiments.async_interface import Network, Scenario, simulate
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_error_budget import audit_updates
from experiments.audit_response_controls import load_model, seal
from experiments.review_async_interface import verify_ledger, gap_comparison


ARMS = dict(baseline=('directional', 'hold'), budget_hold=('error_budget', 'hold'),
    budget_predict=('error_budget', 'response_linear'), model_response=('error_budget', 'model_response'),
    model_frozen=('error_budget', 'model_frozen'))


def at_common_bytes(candidate, reference):
    budget = int(min(candidate['counters'][-1, 0], reference['counters'][-1, 0]))
    a, b = [int(np.searchsorted(run['counters'][:, 0], budget, side='right')-1)
            for run in (candidate, reference)]
    if min(a, b) < 0:
        raise ValueError('no common post-update byte budget')
    cg, rg = float(candidate['relative_gaps'][a]), float(reference['relative_gaps'][b])
    return dict(budget=budget, candidate_tick=a, reference_tick=b, candidate_gap=cg,
                reference_gap=rg, result=gap_comparison(cg, rg))


def audit_bytes(model, links, scenario, run):
    """Replay integer traffic only, without Packet.wire_bytes or Network.send."""
    topology = Network(links, scenario, lambda packet: None)
    model_packets = run['summary']['prediction'] in ('model_response', 'model_frozen')
    ids = sorted(model.indices)
    periods = {i: scenario.period(rank) for rank, i in enumerate(ids)}
    dependencies = [(i, j) for i, row in model.rows().items() for j in row if j != i]
    setup = sum((48+24*len(model.indices[i])**2)*(len(topology.paths[i, j])-1)
                for i, j in dependencies) if model_packets else 0
    total, hops, pending, expected = setup, 0, [], []
    for tick in range(len(run['states'])):
        for i, j in dependencies:
            if tick % periods[i] == 0:
                heapq.heappush(pending, (tick, i, j, 0))
        while pending and pending[0][0] <= tick:
            now, i, j, hop = heapq.heappop(pending)
            path = topology.paths[i, j]
            if hop == len(path)-1:
                continue
            a, b = path[hop:hop+2]
            dimension = len(model.indices[i])
            total += 40+8*dimension+(16+8*dimension if model_packets else 0)
            hops += 1
            can_drop = scenario.name != 'loss' or hop == 0
            if not (can_drop and scenario.lost(a, b, now, topology.outage_edge)):
                heapq.heappush(pending, (now+scenario.latency(a, b, now), i, j, hop+1))
        expected.append((total, sum(tick//periods[i]+1 for i in ids)))
    np.testing.assert_array_equal(run['counters'], expected)
    assert run['summary']['model_setup_bytes'] == setup
    assert run['summary']['hop_messages'] == hops
    return dict(prefix_bytes_exact=True, prefix_updates_exact=True, model_setup_bytes=setup,
                application_bytes=total, hop_messages=hops)


def summarize(rows, comparisons):
    return dict(runs=len(rows), incomplete=sum(not r['horizon_complete'] for r in rows),
        comparisons={name: {metric: {key: sum(c[metric] == key for c in comparisons if c['name'] == name)
            for key in ('win', 'tie', 'loss')} for metric in ('equal_bytes', 'equal_ticks')}
            for name in sorted({c['name'] for c in comparisons})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/dranppr_response_final_20261124')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['d1', 'd4', 'se3_31', 'coupled_spd'])
    parser.add_argument('--scenarios', nargs='+', default=['delay', 'outage'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[34])
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--initial-seed', type=int)
    args = parser.parse_args()
    verify_ledger(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['experiments/'+n+'.py' for n in ('quadratic_response', 'async_interface',
        'async_interface_model', 'prediction_error_budget', 'neighbor_prediction', 'audit_model_response',
        'audit_error_budget', 'audit_response_controls', 'audit_async_interface', 'review_async_interface',
        'audit_history_forecast', 'network_noise_support')]
    sources += ['tests/test_quadratic_response.py', 'docs/superpowers/plans/2026-11-24-model-response-feedback.md']
    for name in sources:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    manifest = dict(command=sys.argv, all_terminal=False, cases=args.cases, scenarios=args.scenarios,
        seeds=args.seeds, ticks=args.ticks, arms=ARMS, stress=args.stress, initial_seed=args.initial_seed,
        source_root=str(args.root.resolve()), source_ledger=digest(args.root/'SHA256SUMS'),
        scope='Fixed quadratics, not native online SE3 or ATE', truth_used=False, memory=8,
        model_setup='Incremental descriptor bytes included; delivery assumed; common offline row setup excluded')
    write_json(args.output/'manifest.json', manifest)
    rows, comparisons = [], []
    for case in args.cases:
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
                runs = {}
                for arm, (mode, prediction) in ARMS.items():
                    started = time.perf_counter()
                    run = simulate(model, links, scenario, mode, args.ticks, initial, prediction=prediction)
                    elapsed = time.perf_counter()-started
                    checked = audit_updates(model, run, initial)
                    traffic = audit_bytes(model, links, scenario, run)
                    optimum = np.linalg.solve(model.b, -model.c)
                    error, start_error = run['states']-optimum, initial-optimum
                    independent_gaps = np.einsum('ij,ij->i', error@model.b, error)/(start_error@model.b@start_error)
                    np.testing.assert_allclose(run['relative_gaps'], independent_gaps, rtol=1e-12, atol=1e-30)
                    if net == 'ideal' and runs:
                        np.testing.assert_array_equal(run['states'], runs['baseline']['states'])
                    if arm == 'model_frozen':
                        np.testing.assert_array_equal(run['counters'], runs['model_response']['counters'])
                    if runs:
                        np.testing.assert_array_equal(run['counters'][:, 1], runs['baseline']['counters'][:, 1])
                    runs[arm] = run
                    row = dict(run['summary'], arm=arm, case=case, service_seconds=elapsed,
                               audit=checked, traffic_audit=traffic)
                    rows.append(row)
                    name = f'{net}_{seed}_{arm}'
                    write_json(folder/(name+'.json'), row)
                    np.savez_compressed(folder/(name+'.npz'), **{k: v for k, v in run.items() if k != 'summary'})
                    print(json.dumps(dict(case=case, scenario=net, seed=seed, arm=arm,
                        gap=row['relative_gap'], bytes=row['application_bytes'], seconds=elapsed)), flush=True)
                for other in ('baseline', 'budget_hold', 'budget_predict', 'model_frozen'):
                    aligned = at_common_bytes(runs['model_response'], runs[other])
                    comparisons.append(dict(case=case, scenario=net, seed=seed,
                        name='model_response_vs_'+other, equal_bytes=aligned['result'], alignment=aligned,
                        equal_ticks=gap_comparison(runs['model_response']['relative_gaps'][-1],
                                                   runs[other]['relative_gaps'][-1])))
        write_json(args.output/'runs.json', rows)
        write_json(args.output/'comparisons.json', comparisons)
        write_json(args.output/'summary.json', summarize(rows, comparisons))
    seal(args.output, manifest)
    print(json.dumps(summarize(rows, comparisons)), flush=True)


if __name__ == '__main__':
    main()
