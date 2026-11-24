"""Independent input/score checks and descriptive rolling-forecast diagnostics."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from experiments.analyze_network_noise_transfer import rescore
from experiments.audit_mean_evidence import sha256, write_json
from experiments.audit_rolling_prediction import ARMS, statistics


def check_weights(nominal, adapted, scales, cutoff):
    left, right = copy.deepcopy(nominal), copy.deepcopy(adapted)
    errors = []
    for package, q in [(left, np.ones(5)), (right, np.asarray(scales))]:
        explicit = package['schema'] == 'dranppr_frozen_joint_v2'
        if not explicit and package['schema'] != 'dranppr_frozen_joint_v1':
            raise ValueError('unknown weight schema')
        package.pop('suffix_variance_multipliers', None)
        for robot in package['robots']:
            times = {p['index']:p['timestamp_s'] for p in robot['poses']}
            for kind in ('odometry', 'observations'):
                for f in robot[kind]:
                    if explicit != ('sigmas' in f):
                        raise ValueError('factor weight/schema mismatch')
                    time = times[f['target' if kind == 'odometry' else 'pose']]
                    multipliers = q if time > cutoff else np.ones(5)
                    base = (np.array([.1, .1, .05])*math.sqrt(f['dt'])
                            if kind == 'odometry' else np.array([.05, .1]))
                    expected = base*np.sqrt(multipliers[:3] if kind == 'odometry' else multipliers[3:])
                    actual = np.asarray(f.pop('sigmas')) if explicit else base
                    errors.append(float(np.max(np.abs(actual-expected))))
    if left != right:
        raise ValueError('non-weight input differs')
    error = max(errors, default=0.)
    if error > 1e-12:
        raise ValueError('incorrect prefix/suffix weight')
    return error


def check_prediction(row, event):
    innovation = np.array([event['bearing_rad']-row['mean'][0], event['range_m']-row['mean'][1]])
    innovation[0] = math.atan2(math.sin(innovation[0]), math.cos(innovation[0]))
    cov = np.asarray(row['covariance'])
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0.:
        raise ValueError('nonpositive predictive covariance')
    energy = float(innovation@np.linalg.solve(cov, innovation))
    nll = .5*(energy+np.linalg.slogdet(cov)[1]+2*np.log(2*np.pi))
    errors = dict(innovation=float(np.max(np.abs(innovation-row['innovation']))),
                  energy=abs(energy-row['energy']), nll=abs(nll-row['nll']))
    if max(errors.values()) > 1e-8:
        raise ValueError('measurement reconstruction or predictive score mismatch')
    return errors


def analyze(output):
    manifest = json.loads((output/'manifest.json').read_text())
    if not manifest['all_workers_terminal']:
        raise ValueError('incomplete rolling experiment')
    source = Path(manifest['source'])
    for name, digest in manifest['source_hashes'].items():
        if sha256(source/name) != digest:
            raise ValueError('source archive changed: '+name)
    for name, digest in {**manifest['library_hashes'], **manifest['binary_hashes']}.items():
        if sha256(Path(name)) != digest:
            raise ValueError('runtime changed: '+name)
    maximum = dict(weight=0., innovation=0., energy=0., nll=0., covariance_residual=0., old_parity=0.)
    old_controls, solves, commands, groups, swaps = 0, [], [], [], []
    checked_predictions = 0
    for dataset in (1, 4):
        config = manifest['datasets'][str(dataset)]
        first = config['first_timestamp_s']
        events = [json.loads(s) for s in (source/f'd{dataset}/events.jsonl').read_text().splitlines()]
        lookup = {(e['robot'], e['source_row']):e for e in events if e['kind']=='observation'}
        for index, (start, end) in enumerate(config['windows']):
            folder = output/f'd{dataset}_w{index:03d}'
            packages = {a:json.loads((folder/(a+'.json')).read_text()) for a in ARMS}
            maximum['weight'] = max(maximum['weight'], check_weights(
                packages['nominal'], packages['joint_qr'], config['scales'], first+config['cutoff_elapsed_s']))
            for robot in packages['nominal']['robots']:
                times = {p['index']:p['timestamp_s'] for p in robot['poses']}
                if max(times.values()) > first+start:
                    raise ValueError('future pose in fitted graph')
                for obs in robot['observations']:
                    e = lookup[robot['id'], obs['source']]
                    if (e['timestamp_s'] != times[obs['pose']] or e['range_m'] != obs['range'] or
                            e['bearing_rad'] != obs['bearing']):
                        raise ValueError('past physical observation changed')
            verification = json.loads((folder/'verification.json').read_text())
            maximum['covariance_residual'] = max(maximum['covariance_residual'],
                                                 *verification['covariance_residuals'].values())
            old_controls += len(verification['old_window_parity'])
            for row in verification['old_window_parity'].values():
                maximum['old_parity'] = max(maximum['old_parity'], *row.values())
            commands.extend(json.loads((folder/'commands.json').read_text()))
            for arm in ARMS:
                solves.append(json.loads((folder/arm/'summary.json').read_text()))
        rows = {a:json.loads((output/f'd{dataset}_{a}_predictions.json').read_text()) for a in ARMS}
        for arm in ARMS:
            for row in rows[arm]:
                e = lookup[row['robot'], row['source_row']]
                if e['timestamp_s'] != row['timestamp_s'] or e['landmark_subject'] != row['subject']:
                    raise ValueError('prediction event identity mismatch')
                for key, value in check_prediction(row, e).items():
                    maximum[key] = max(maximum[key], value)
                checked_predictions += 1
            for index in range(len(config['windows'])):
                for robot in range(1, 6):
                    selected = [r for r in rows[arm] if r['window']==index and r['robot']==robot]
                    groups.append(dict(dataset=dataset, arm=arm, window=index, robot=robot,
                                       **statistics(selected)))
        for mean in ARMS:
            for covariance in ARMS:
                scores = rescore(rows[mean], rows[covariance])
                swaps.append(dict(dataset=dataset, mean_arm=mean, covariance_arm=covariance,
                                  nll=float(np.mean([s['nll'] for s in scores]))))
    if old_controls != 12 or any(r['exit_code'] != 0 for r in commands):
        raise ValueError('missing controls or failed native process')
    result = dict(maximum_errors=maximum, predictions_checked=checked_predictions,
        old_forecast_controls=old_controls, native_commands=len(commands),
        nonlinear_solves=len(solves), max_lm_iterations=max(s['lm_iterations'] for s in solves),
        max_posthoc_gradient=max(s['posthoc_gradient_norm'] for s in solves),
        worker_seconds=sum(r['seconds'] for r in manifest['runs']),
        max_worker_seconds=max(r['seconds'] for r in manifest['runs']),
        source_files_verified=len(manifest['source_hashes']), truth_used=False,
        groups=groups, mean_covariance_swaps=swaps,
        caveat='Mean/covariance swaps are post-selected descriptive attributions, not alternative fits. All grouped measurements are dependent; no significance test or effective sample count claimed.')
    write_json(output/'verification_analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'groups'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    analyze(parser.parse_args().output)
