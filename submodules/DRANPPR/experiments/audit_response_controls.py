"""Forecast, stress, and measured-compute controls for response prediction."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from experiments.async_interface import Scenario, simulate, Network, delay_bounds, directional_majorizer
from experiments.async_interface_model import NormalizedModel, normalize
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_history_forecast import trajectory
from experiments.neighbor_prediction import HistoryPrediction, ResponsePrediction
from experiments.review_async_interface import verify_ledger


MODES = ('hold', 'secant', 'risk3', 'response_linear', 'response_calibrated')


def predictor(mode):
    return ResponsePrediction(mode) if mode.startswith('response_') else HistoryPrediction(mode)


def load_model(root, case):
    data = np.load(root/case/'quadratic.npz')
    metadata = json.loads((root/case/'model.json').read_text())
    model = NormalizedModel(data['h'], data['g'], data['b'], data['c'], data['transform'],
        {int(i): np.array(ii) for i, ii in metadata['indices'].items()}, float(np.linalg.cond(data['b'])))
    return model, metadata['links']


def exact_neighbor_diagnostic(model, links, scenario, ticks, fresh_metric=False):
    """Unimplementable control: current global state is granted for free."""
    ids = sorted(model.indices)
    periods = {i: scenario.period(k) for k, i in enumerate(ids)}
    metric_scenario = Scenario('ideal') if fresh_metric else scenario
    network = Network(links, metric_scenario, lambda packet: None)
    bounds = delay_bounds(network, ids, periods)
    steps = {i: np.linalg.inv(directional_majorizer(row, i, bounds)) for i, row in model.rows().items()}
    initial = np.sin(np.arange(len(model.c))+1.)
    value = initial.copy()
    states = []
    for tick in range(ticks):
        gradient = model.b@value+model.c
        for i, indices in model.indices.items():
            if tick % periods[i] == 0:
                value[indices] -= steps[i]@gradient[indices]
        states.append(value.copy())
    optimum = np.linalg.solve(model.b, -model.c)
    error = np.asarray(states)-optimum
    start = initial-optimum
    gaps = np.einsum('ij,ij->i', error@model.b, error)/(start@model.b@start)
    return dict(states=np.asarray(states), relative_gaps=gaps, deployable=False,
        fresh_metric=fresh_metric, relative_gap=float(gaps[-1]))


def forecasts(trials):
    rows = []
    times = np.array([-14., -12., -9., -7., -6., -3., -1., 0.])
    modes = MODES+('poly1',)
    for curve in ('linear', 'quadratic', 'relaxation', 'stop', 'reverse'):
        for noise in (0., .01, .05):
            observed = np.array([trajectory(curve, t) for t in times])[None, :, :]
            samples = observed+np.random.default_rng(31).normal(0., noise, (trials, len(times), 2))
            errors = {(m, h): [] for m in modes for h in (1., 4., 16.)}
            for values in samples:
                for mode in modes:
                    p = predictor(mode)
                    for t, value in zip(times, values):
                        p.add(t, value)
                    for horizon in (1., 4., 16.):
                        error = p.predict(horizon)-trajectory(curve, horizon)
                        errors[mode, horizon].append(float(error@error))
            for (mode, horizon), values in errors.items():
                rows.append(dict(curve=curve, noise=noise, horizon=horizon, prediction=mode,
                    trials=trials, mean_squared_error=float(np.mean(values)),
                    q95_squared_error=float(np.quantile(values, .95))))
    # A new event is not inferable from an identical history. Recovery begins
    # only after the changed source state is actually observed.
    recovery = []
    for mode in modes:
        p = predictor(mode)
        for t in range(-16, 1):
            p.add(t, trajectory('linear', t))
        for t in range(13):
            if t:
                p.add(t, trajectory('stop', t))
            error = p.predict(t+4)-trajectory('stop', t+4)
            recovery.append(dict(prediction=mode, received_post_change=t,
                squared_error=float(error@error), gain=getattr(p, 'gain', None)))
    return dict(cells=rows, stop_recovery=recovery,
        scope='Stylized coordinate histories, clean future reference; not SLAM sensor-noise robustness')


def stress(root, output):
    rows = []
    for case in ('d1', 'd4', 'se3_31', 'coupled_spd'):
        model, links = load_model(root, case)
        if case == 'coupled_spd':
            links = [(i, j) for i in range(5) for j in range(i+1, 5)]
        for name, scenario in (
                ('long_delay', Scenario('delay', 12, 20, seed=34)),
                ('long_outage', Scenario('outage', 1, 5, seed=34, outage=(200, 600)))):
            reference = None
            for mode in MODES:
                result = simulate(model, links, scenario, 'directional', 2000,
                    np.sin(np.arange(len(model.c))+1.), prediction=mode)
                if reference is None:
                    reference = result
                np.testing.assert_array_equal(result['counters'], reference['counters'])
                row = dict(result['summary'], case=case, stress=name)
                rows.append(row)
                np.savez_compressed(output/f'{case}_{name}_{mode}.npz',
                    **{k: result[k] for k in ('states', 'relative_gaps', 'counters', 'gradient_error_sq')})
                print(json.dumps(dict(stage='stress', case=case, network=name, prediction=mode,
                    gap=row['relative_gap'], complete=row['horizon_complete'])), flush=True)
    return rows


def benchmark(root, repetitions, ticks):
    rows = []
    # Deterministic shuffled order each repetition, same host and code. Timings
    # include the simulator and posthoc diagnostics, NOT distributed robot CPU.
    rng = np.random.default_rng(20261124)
    for case in ('d1', 'd4', 'se3_31'):
        model, links = load_model(root, case)
        initial = np.sin(np.arange(len(model.c))+1.)
        simulate(model, links, Scenario('delay'), 'directional', 20, initial)
        for network in ('delay', 'heterogeneous'):
            for repeat in range(repetitions):
                for mode in rng.permutation(MODES):
                    started = time.perf_counter()
                    result = simulate(model, links, Scenario(network, seed=31), 'directional', ticks,
                        initial, prediction=mode)
                    elapsed = time.perf_counter()-started
                    rows.append(dict(case=case, scenario=network, repeat=repeat, prediction=str(mode),
                        ticks=ticks, service_seconds=elapsed, relative_gap=result['summary']['relative_gap']))
            print(json.dumps(dict(stage='timing', case=case, network=network)), flush=True)
    return rows


def target_benchmark(root, archive, repetitions):
    """Replay the first archived 1e-6 crossing; stopping is an offline oracle."""
    rows = []
    model, links = load_model(root, 'se3_31')
    initial = np.sin(np.arange(len(model.c))+1.)
    rng = np.random.default_rng(20261124)
    for network in ('delay', 'heterogeneous', 'loss'):
        traces = {mode: np.load((root if mode.startswith('response_') else archive)/'se3_31'/
            f'{network}_31_{mode}.npz') for mode in MODES}
        for repeat in range(repetitions):
            for mode in rng.permutation(MODES):
                hits = np.flatnonzero(traces[mode]['relative_gaps'] < 1e-6)
                if not len(hits):
                    raise ValueError('target not attained by every method')
                ticks = int(hits[0])+1
                started = time.perf_counter()
                result = simulate(model, links, Scenario(network, seed=31), 'directional', ticks,
                    initial, prediction=mode)
                elapsed = time.perf_counter()-started
                np.testing.assert_array_equal(result['states'], traces[mode]['states'][:ticks])
                assert result['relative_gaps'][-1] < 1e-6
                rows.append(dict(case='se3_31', scenario=network, prediction=str(mode), repeat=repeat,
                    ticks=ticks, service_seconds=elapsed, relative_gap=float(result['relative_gaps'][-1]),
                    bytes=int(result['counters'][-1, 0]), archive_state_parity=True))
        print(json.dumps(dict(stage='target timing', case='se3_31', network=network)), flush=True)
    return rows


def seal(output, manifest):
    manifest['all_terminal'] = True
    write_json(output/'manifest.json', manifest)
    paths = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timing-ticks', type=int, default=600)
    parser.add_argument('--target-only-archive', type=Path)
    args = parser.parse_args()
    verify_ledger(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    files = ('experiments/neighbor_prediction.py', 'experiments/async_interface.py',
        'experiments/prediction_error_budget.py',
        'experiments/async_interface_model.py', 'experiments/audit_response_controls.py',
        'experiments/audit_history_forecast.py', 'experiments/audit_async_interface.py',
        'experiments/review_async_interface.py', 'tests/test_response_prediction.py',
        'docs/superpowers/plans/2026-11-24-response-prediction.md')
    for name in files:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    manifest = dict(command=sys.argv, all_terminal=False, source_root=str(args.root.resolve()),
        source_ledger_sha256=digest(args.root/'SHA256SUMS'), platform=platform.platform(),
        python=platform.python_version(), numpy=np.__version__, trials=args.trials,
        repetitions=args.repetitions, timing_ticks=args.timing_ticks, modes=MODES,
        note='One fixed size=8, no endpoint tuning. No native or nonlinear performance claim.')
    write_json(args.output/'manifest.json', manifest)
    if args.target_only_archive:
        verify_ledger(args.target_only_archive)
        manifest['target_archive'] = str(args.target_only_archive.resolve())
        manifest['target_archive_ledger'] = digest(args.target_only_archive/'SHA256SUMS')
        manifest['note'] = ('Measured replay to first relative-gap 1e-6 crossing; '
            'offline stopping oracle, Python simulator not native robot CPU')
        write_json(args.output/'target_timing.json', target_benchmark(args.root,
            args.target_only_archive, args.repetitions))
        seal(args.output, manifest)
        return
    write_json(args.output/'forecasts.json', forecasts(args.trials))
    print('forecast controls complete', flush=True)
    exact = []
    for case in ('d1', 'd4', 'se3_31'):
        model, links = load_model(args.root, case)
        for name in ('delay', 'heterogeneous', 'outage'):
            for fresh in (False, True):
                result = exact_neighbor_diagnostic(model, links, Scenario(name), 2000, fresh)
                exact.append(dict(case=case, scenario=name, fresh_metric=fresh,
                    relative_gap=result['relative_gap'], deployable=False))
                np.savez_compressed(args.output/f'{case}_{name}_exact_{fresh}.npz',
                    states=result['states'], relative_gaps=result['relative_gaps'])
    write_json(args.output/'exact_neighbor_diagnostic.json', exact)
    write_json(args.output/'stress.json', stress(args.root, args.output))
    write_json(args.output/'timing.json', benchmark(args.root, args.repetitions, args.timing_ticks))
    seal(args.output, manifest)


if __name__ == '__main__':
    main()
