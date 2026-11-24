"""Measured prefix replays at a baseline-defined target, not online stopping."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from experiments.async_interface import Scenario, simulate
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_error_budget import ARMS
from experiments.audit_response_controls import load_model, seal
from experiments.review_async_interface import verify_ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cases', nargs='+', default=['d1', 'd4', 'se3_31'])
    parser.add_argument('--scenarios', nargs='+', default=['delay', 'heterogeneous', 'outage'])
    parser.add_argument('--target', choices=['baseline_endpoint', 'fixed_1e6', 'post_outage_endpoint'],
                        default='baseline_endpoint')
    args = parser.parse_args()
    manifest = verify_ledger(args.root)
    if manifest.get('stress') or manifest.get('initial_seed') is not None or manifest['memory'] != 8:
        parser.error('timing control requires the fixed original protocol')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ['baseline', 'budget_hold', 'budget_predict']
    rng = np.random.default_rng(20261124)
    rows, summary = [], []
    for case in args.cases:
        model, links = load_model(args.root, case)
        initial = np.sin(np.arange(len(model.c))+1.)
        for network in args.scenarios:
            archives = {arm: dict(np.load(args.root/case/f'{network}_32_{arm}.npz')) for arm in names}
            target = (1e-6 if args.target == 'fixed_1e6' else
                      float(archives['baseline']['relative_gaps'][-1])*(1.+1e-12))
            earliest = Scenario(network).outage[1] if args.target == 'post_outage_endpoint' else 0
            if args.target == 'post_outage_endpoint' and network != 'outage':
                raise ValueError('post-outage timing requires the outage network')
            prefixes = {}
            for arm in names:
                gap = archives[arm]['relative_gaps']
                crossing = np.flatnonzero((gap <= target) & (np.arange(len(gap)) >= earliest))
                prefixes[arm] = int(crossing[0])+1 if len(crossing) else None
            for arm in names:
                mode, prediction = ARMS[arm]
                simulate(model, links, Scenario(network, seed=32), mode, 20, initial, prediction=prediction)
            for repeat in range(args.repetitions):
                for arm in rng.permutation(names):
                    ticks = prefixes[arm]
                    if ticks is None:
                        rows.append(dict(case=case, scenario=network, arm=str(arm), repeat=repeat,
                                         target=target, attained=False))
                        continue
                    mode, prediction = ARMS[arm]
                    started = time.perf_counter()
                    run = simulate(model, links, Scenario(network, seed=32), mode, ticks, initial,
                                   prediction=prediction)
                    elapsed = time.perf_counter()-started
                    np.testing.assert_array_equal(run['states'], archives[arm]['states'][:ticks])
                    np.testing.assert_array_equal(run['counters'], archives[arm]['counters'][:ticks])
                    assert run['relative_gaps'][-1] <= target
                    rows.append(dict(case=case, scenario=network, arm=str(arm), repeat=repeat,
                        target=target, attained=True, ticks=ticks, seconds=elapsed,
                        application_bytes=int(run['counters'][-1, 0]), archive_prefix_parity=True))
            selected = [r for r in rows if r['case'] == case and r['scenario'] == network and r['attained']]
            medians = {arm: float(np.median([r['seconds'] for r in selected if r['arm'] == arm]))
                       for arm in names if any(r['arm'] == arm for r in selected)}
            records = {r['arm']: r for r in selected}
            for arm in names[1:]:
                if arm in records and 'baseline' in records:
                    summary.append(dict(case=case, scenario=network, arm=arm, target=target,
                        candidate_ticks=prefixes[arm], baseline_ticks=prefixes['baseline'],
                        service_time_ratio=medians[arm]/medians['baseline'],
                        candidate_seconds=medians[arm], baseline_seconds=medians['baseline'],
                        byte_ratio=records[arm]['application_bytes']/records['baseline']['application_bytes']))
            print(json.dumps(dict(case=case, network=network, median_seconds=medians, prefixes=prefixes)), flush=True)
            write_json(args.output/'runs.json', rows)
            write_json(args.output/'summary.json', summary)
    for name in ('benchmark_error_budget.py', 'async_interface.py', 'prediction_error_budget.py',
            'neighbor_prediction.py', 'async_interface_model.py', 'audit_error_budget.py',
            'audit_async_interface.py', 'audit_neighbor_prediction.py', 'audit_response_controls.py',
            'audit_history_forecast.py', 'network_noise_support.py', 'review_async_interface.py'):
        target = args.output/'source'/'experiments'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name, target)
    seal(args.output, dict(root=str(args.root.resolve()), root_ledger=digest(args.root/'SHA256SUMS'),
        target=args.target, cases=args.cases, scenarios=args.scenarios,
        stopping='Offline oracle first crossing; post_outage requires a tick after the blackout ends',
        seed=32, repetitions=args.repetitions, platform=platform.platform(), numpy=np.__version__,
        scope='Same-process Python simulator plus diagnostics; not native distributed runtime',
        random_order=True, setup_bytes_included=False))


if __name__ == '__main__':
    main()
