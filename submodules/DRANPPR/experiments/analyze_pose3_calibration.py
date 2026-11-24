"""Describe all declared SE3 controls, including unchanged-noise regressions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.audit_mean_evidence import sha256, write_json


def paired_summary(nominal, calibrated, oracle):
    arrays = [np.asarray(x,float) for x in (nominal,calibrated,oracle)]
    if (arrays[0].ndim != 1 or len(arrays[0]) < 2 or
            any(x.shape != arrays[0].shape or not np.isfinite(x).all() or
                np.any(x < 0.) for x in arrays) or arrays[0].mean() <= 0.):
        raise ValueError('invalid paired errors')
    a,b,c = arrays
    return dict(means={k:float(x.mean()) for k,x in zip(('nominal','calibrated','oracle'),arrays)},
        standard_deviations={k:float(x.std(ddof=1)) for k,x in zip(('nominal','calibrated','oracle'),arrays)},
        calibrated_wins=int(np.sum(b<a)),calibrated_losses=int(np.sum(b>a)),ties=int(np.sum(b==a)),
        mean_reduction_percent=float(100.*(1.-b.mean()/a.mean())),
        max_paired_increase=float((b-a).max()))


def analyze(source, output):
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('changed producer archive: '+name)
    manifest = json.loads((source/'manifest.json').read_text())
    rows = json.loads((source/'comparison.json').read_text())
    expected = {(regime,seed) for regime in manifest['regimes'] for seed in manifest['seeds']}
    if (not manifest['all_terminal'] or len(rows) != len(expected) or
            {(r['regime'],r['seed']) for r in rows} != expected or
            len(manifest['runs']) != 3*len(rows) or any(r['exit_code'] != 0 for r in manifest['runs'])):
        raise ValueError('incomplete declared SE3 support')
    metrics = tuple(rows[0]['nominal']['quality'])
    for row in rows:
        case = source/f"{row['regime']}_{row['seed']}"
        for name in ('training.json','test_input.json'):
            if len({sha256(case/arm/name) for arm in ('nominal','calibrated','oracle')}) != 1:
                raise ValueError('changed physical input')
        if row['regime'] == 'nominal' and row['nominal']['quality'] != row['oracle']['quality']:
            raise ValueError('nominal oracle differs from nominal model')
    summary = dict(source=str(source),source_ledger_sha256=sha256(source/'SHA256SUMS'),
        scope=manifest['scope'],regimes={},
        max_jacobian_error=max(r['nominal']['jacobian_max_error'] for r in rows),
        max_nll_parity_error=max(r['parity']['nll_abs_error'] for r in rows),
        max_score_parity_error=max(r['parity']['score_max_error'] for r in rows),
        interpretation='Paired fixed-seed descriptive statistics, one geometry. No universal safety, online, real-data or outlier robustness claim.')
    for regime in manifest['regimes']:
        selected = [r for r in rows if r['regime']==regime]
        summary['regimes'][regime] = dict(count=len(selected),metrics={m:paired_summary(
            *[[r[arm]['quality'][m] for r in selected] for arm in ('nominal','calibrated','oracle')])
            for m in metrics},boundary_seeds=[r['seed'] for r in selected if r['fit']['at_numerical_boundary']],
            median_fitted_scales=np.median([r['fit']['components'] for r in selected],axis=0).tolist())
    if output.exists():
        raise ValueError('analysis output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.source.resolve(),args.output.resolve())
