"""Post-selected forecast attribution, without fitting or changing an estimator."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.audit_lazy_refresh import dump
from experiments.audit_network_noise_transfer import prediction_summary


def rescore(mean_rows, covariance_rows):
    if len(mean_rows) != len(covariance_rows):
        raise ValueError('different predictive support')
    output = []
    for left, right in zip(mean_rows, covariance_rows):
        identity = ('robot','source_row','timestamp_s','subject')
        if any(left[k] != right[k] for k in identity):
            raise ValueError('different predictive support')
        measurement_difference = (np.array(left['mean'])+left['innovation']-
                                  np.array(right['mean'])-right['innovation'])
        measurement_difference[0] = math.atan2(math.sin(measurement_difference[0]), math.cos(measurement_difference[0]))
        if np.linalg.norm(measurement_difference) > 1e-9:
            raise ValueError('different observations')
        innovation = np.array(left['innovation'])
        chol = np.linalg.cholesky(right['covariance'])
        whitened = np.linalg.solve(chol, innovation)
        energy = float(whitened@whitened)
        nll = .5*(energy+2*np.log(np.diag(chol)).sum()+2*np.log(2*np.pi))
        output.append(dict(energy=energy,nll=float(nll)))
    return output


def moments(rows):
    innovations = np.array([r['innovation'] for r in rows])
    covariance = np.array([r['covariance'] for r in rows])
    whitened = np.array([np.linalg.solve(np.linalg.cholesky(c), v)
                        for v,c in zip(innovations,covariance)])
    return dict(count=len(rows), innovation_mean=innovations.mean(axis=0).tolist(),
        innovation_centered_sd=innovations.std(axis=0).tolist(),
        whitened_mean=whitened.mean(axis=0).tolist(),
        mean_predicted_sd=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2)).mean(axis=0).tolist(),
        energy_quantiles=np.quantile([r['energy'] for r in rows],[.5,.9,.95,.99]).tolist(),
        log_volume_term=float(np.linalg.slogdet(covariance)[1].mean()/2),
        **{k:v for k,v in prediction_summary(rows).items() if k != 'count'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output/'manifest.json').read_text())
    if not manifest['all_predictions_terminal']:
        raise ValueError('incomplete forecasts')
    records = []
    for dataset in (1,4):
        arms = {}
        for arm in ('nominal','joint_qr'):
            windows = {t:json.loads((args.output/f'd{dataset}/t{t}_{arm}/forecast.json').read_text())['predictions']
                       for t in (30,60,90)}
            arms[arm] = [r for rows in windows.values() for r in rows]
            for robot in sorted({r['robot'] for r in arms[arm]}):
                for elapsed, rows in [(None,arms[arm]),*windows.items()]:
                    selected = [r for r in rows if r['robot']==robot]
                    if selected:
                        records.append(dict(dataset=dataset,arm=arm,robot=robot,elapsed=elapsed,**moments(selected)))
        for mean_arm in arms:
            for covariance_arm in arms:
                records.append(dict(dataset=dataset,mean_arm=mean_arm,covariance_arm=covariance_arm,
                    **prediction_summary(rescore(arms[mean_arm],arms[covariance_arm]))))
    dump(args.output/'prediction_attribution.json', dict(records=records,
        post_selected=True, truth_used=False, refit=False,
        scope='Descriptive robot/window moments and mean/covariance swaps. No independence assumption, statistical test, alternative estimator, hyperparameter selection or pure-sensor-noise interpretation.'))
    for row in records:
        if 'mean_arm' in row or (row['arm']=='joint_qr' and row['elapsed'] is None):
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
