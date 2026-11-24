"""Stylized noisy-history and unknowable-change controls, not SLAM accuracy."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from experiments.neighbor_prediction import HistoryPrediction
from experiments.audit_async_interface import ROOT, digest, write_json


def trajectory(name, time):
    velocity = np.array([.02,.01])
    if name == 'linear':
        return time*velocity
    if name == 'quadratic':
        return time*velocity+time*time*np.array([.0002,-.0001])
    if name == 'relaxation':
        return np.array([1.,.4])*(1.-np.exp(-np.array([.03,.1])*(time+20.)))
    if name == 'stop':
        return min(time,0.)*velocity
    if name == 'reverse':
        return -abs(time)*velocity
    raise ValueError('unknown trajectory')


def forecast_controls(trials=200, noises=(0.,.01,.05), horizons=(1.,4.,16.)):
    times = np.array([-14.,-12.,-9.,-7.,-6.,-3.,-1.,0.])
    modes = ('hold','secant','poly1','poly2','poly3','risk3')
    rows, errors = [], {}
    for curve in ('linear','quadratic','relaxation','stop','reverse'):
        for noise in noises:
            rng = np.random.default_rng(31)
            observed = np.array([trajectory(curve,t) for t in times])[None,:,:]+rng.normal(0.,noise,(trials,8,2))
            for horizon in horizons:
                for mode in modes:
                    values, degrees = [], np.zeros(4,dtype=np.int64)
                    for samples in observed:
                        predictor = HistoryPrediction(mode,8)
                        for t,sample in zip(times,samples):
                            predictor.add(t,sample)
                        error = predictor.predict(horizon)-trajectory(curve,horizon)
                        values.append(float(error@error))
                        degrees += predictor.degree_counts
                    name = f'{curve}_{noise}_{horizon}_{mode}'
                    errors[name] = np.asarray(values)
                    rows.append(dict(curve=curve, noise=noise, horizon=horizon, prediction=mode,
                        trials=trials, mean_squared_error=float(np.mean(values)),
                        median_squared_error=float(np.median(values)),
                        q95_squared_error=float(np.quantile(values,.95)), degree_counts=degrees.tolist()))
    return rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials',type=int,default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('experiments/audit_history_forecast.py','experiments/neighbor_prediction.py',
                 'tests/test_history_forecast_controls.py','tests/test_neighbor_prediction.py'):
        target = args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    rows, errors = forecast_controls(args.trials)
    write_json(args.output/'manifest.json',dict(seed=31,trials=args.trials,all_terminal=True,
        scope='Independent synthetic history prediction, not optimizer or sensor-noise robustness',
        source_times=[-14.,-12.,-9.,-7.,-6.,-3.,-1.,0.],noises=[0.,.01,.05],horizons=[1.,4.,16.],
        information='All predictors see identical past samples only; stop/reverse occur after last sample'))
    write_json(args.output/'summary.json',rows)
    np.savez_compressed(args.output/'squared_errors.npz',**errors)
    files = sorted(p for p in args.output.rglob('*') if p.is_file())
    (args.output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(args.output)}\n' for p in files))
    for row in rows:
        if row['curve']=='quadratic' and row['noise']==.01 and row['horizon']==16.:
            print(row,flush=True)


if __name__ == '__main__':
    main()
