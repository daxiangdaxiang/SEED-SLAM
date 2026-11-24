"""Matched, GT-free range attribution. Swapped predictions are diagnostics only."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import chi2

from experiments.analyze_motion_time_structure import split_score
from experiments.audit_lazy_refresh import dump
from experiments.audit_motion_time_structure import verify_archive


SCORE_KEYS=('joint','bearing','range_given_bearing','range','bearing_given_range',
            'range_energy','range_log_volume','range_coverage95')
RANGE95=float(chi2.ppf(.95,1))


def prediction_key(p):
    return tuple(p[k] for k in ('robot','source_row','subject','timestamp_s'))


def matched_predictions(nominal, adapted):
    maps=[]
    for forecast in (nominal,adapted):
        if forecast.get('truth_used') is not False:
            raise ValueError('truth-assisted or undeclared forecast')
        rows=forecast['predictions'];index={prediction_key(p):p for p in rows}
        if len(index)!=len(rows):
            raise ValueError('duplicate prediction')
        maps.append(index)
    if not maps[0] or maps[0].keys()!=maps[1].keys():
        raise ValueError('prediction support differs')
    return [(maps[0][k],maps[1][k]) for k in sorted(maps[0])]


def audit_forecast_support(forecast, package, events):
    if forecast.get('truth_used') is not False or package.get('truth_used') is not False:
        raise ValueError('truth-assisted support')
    checkpoint=forecast['checkpoint_s'];end=checkpoint+forecast['horizon_s']
    known={(r['id'],o['label']) for r in package['robots'] for o in r['objects']}
    if any(p['timestamp_s']>checkpoint for r in package['robots'] for p in r['poses']):
        raise ValueError('future state in checkpoint')
    predicted={prediction_key(p):p for p in forecast['predictions']}
    if len(predicted)!=len(forecast['predictions']):
        raise ValueError('duplicate forecast support')
    expected={};unknown=[];last=checkpoint;largest=0.
    for e in events:
        t=e['timestamp_s']
        if not checkpoint<t<=end:
            continue
        if t<last or e['available_time_s']!=t or e['support_max_time_s']>t:
            raise ValueError('future or unordered event support')
        last=t
        if e['kind']!='observation':
            continue
        key=(e['robot'],e['source_row'],e['landmark_subject'],t)
        if key in expected or key in unknown:
            raise ValueError('duplicate source observation')
        if (e['robot'],e['landmark_subject']) in known:
            expected[key]=e
        else:
            unknown.append(key)
    if predicted.keys()!=expected.keys() or len(unknown)!=forecast['unknown_object_observations']:
        raise ValueError('source/forecast support mismatch')
    for key,p in predicted.items():
        e=expected[key];value=np.array(p['mean'])+p['innovation']
        difference=value-np.array([e['bearing_rad'],e['range_m']])
        difference[0]=math.atan2(math.sin(difference[0]),math.cos(difference[0]))
        largest=max(largest,float(np.max(np.abs(difference))))
    if largest>1e-10:
        raise ValueError('source observation differs')
    return dict(known=len(expected),unknown=len(unknown),total=len(expected)+len(unknown),
        unknown_keys=unknown,max_observation_difference=largest)


def paired_prediction(nominal, adapted):
    if prediction_key(nominal)!=prediction_key(adapted):
        raise ValueError('prediction support differs')
    predictions=(nominal,adapted);innovation=[];state_cov=[];measurement_cov=[];observed=[]
    endpoint={};parity=0.
    for name,p in zip(('nominal','adapted'),predictions):
        e=np.asarray(p['innovation'],float);c=np.asarray(p['covariance'],float)
        r=np.asarray(p['observation_covariance'],float);mu=np.asarray(p['mean'],float)
        if e.shape!=(2,) or mu.shape!=(2,) or not np.isfinite(mu).all():
            raise ValueError('invalid observation mean')
        for matrix in (c,r):
            if (matrix.shape!=(2,2) or not np.isfinite(matrix).all() or
                    not np.allclose(matrix,matrix.T,rtol=1e-12,atol=1e-14) or
                    np.linalg.eigvalsh(matrix).min()<=0):
                raise ValueError('invalid predictive/measurement covariance')
        u=c-r
        if np.linalg.eigvalsh(u).min() < -1e-12*max(1.,np.linalg.norm(c)):
            raise ValueError('negative state/process covariance')
        score=split_score(e,c)
        error=abs(score['joint']-p['nll'])
        if not math.isfinite(error) or error>1e-10:
            raise ValueError('archived score differs')
        parity=max(parity,error)
        observed.append(mu+e);innovation.append(e);state_cov.append(u);measurement_cov.append(r)
        endpoint[name]=dict(range_innovation=float(e[1]),range_variance=float(c[1,1]),
            range_state_process_variance=float(u[1,1]),range_measurement_variance=float(r[1,1]),
            bearing_range_correlation=float(c[0,1]/np.sqrt(c[0,0]*c[1,1])))
    difference=observed[1]-observed[0]
    difference[0]=math.atan2(math.sin(difference[0]),math.cos(difference[0]))
    if np.max(np.abs(difference))>1e-10:
        raise ValueError('reconstructed observation differs')
    scores={}
    for m in (0,1):
        for u in (0,1):
            for r in (0,1):
                c=state_cov[u]+measurement_cov[r];e=innovation[m]
                s=split_score(e,c)
                energy=float(e[1]**2/c[1,1])
                s.update(range_energy=energy,range_log_volume=float(.5*np.log(2*np.pi*c[1,1])),
                    range_coverage95=float(energy<=RANGE95))
                scores[f'm{m}_u{u}_r{r}']=s
    return dict(**{k:nominal[k] for k in ('robot','source_row','subject','timestamp_s','horizon_s')},
        scores=scores,endpoints=endpoint,max_score_parity=parity,
        max_observation_difference=float(np.max(np.abs(difference))))


def summarize_rows(rows):
    if not rows:
        raise ValueError('empty summary')
    frames=defaultdict(list)
    for r in rows:
        frames[r['dataset'],r['elapsed'],r['robot'],r['timestamp_s']].append(r)
    cells=sorted(rows[0]['scores'])
    scores={cell:{k:float(np.mean([r['scores'][cell][k] for r in rows])) for k in SCORE_KEYS}
            for cell in cells}
    frame_scores={cell:{k:float(np.mean([np.mean([r['scores'][cell][k] for r in f])
        for f in frames.values()])) for k in SCORE_KEYS} for cell in cells}
    l00,l10,l01,l11=[scores[c]['range'] for c in ('m0_u0_r0','m1_u0_r0','m0_u1_r1','m1_u1_r1')]
    effects=dict(total=l11-l00,mean_at_old_covariance=l10-l00,covariance_at_old_mean=l01-l00,
        mean_at_new_covariance=l11-l01,covariance_at_new_mean=l11-l10,
        interaction=l11-l10-l01+l00)
    endpoints={}
    for name in ('nominal','adapted'):
        e=np.array([r['endpoints'][name]['range_innovation'] for r in rows])
        endpoints[name]={k:float(np.mean([r['endpoints'][name][k] for r in rows]))
                         for k in rows[0]['endpoints'][name]}
        endpoints[name].update(range_rmse=float(np.sqrt(np.mean(e**2))),range_centered_sd=float(np.std(e)))
    delta=np.array([r['scores']['m1_u1_r1']['range']-r['scores']['m0_u0_r0']['range'] for r in rows])
    frame_delta=[np.mean([r['scores']['m1_u1_r1']['range']-r['scores']['m0_u0_r0']['range'] for r in f])
                 for f in frames.values()]
    harm=dict(positive_mass=float(np.maximum(delta,0).sum()),negative_mass=float(-np.minimum(delta,0).sum()),
        worsened_observations=int(sum(delta>0)),improved_observations=int(sum(delta<0)),
        worsened_frames=int(sum(d>0 for d in frame_delta)),improved_frames=int(sum(d<0 for d in frame_delta)))
    return dict(observations=len(rows),frames=len(frames),scores=scores,frame_equal_scores=frame_scores,
        range_effects=effects,endpoints=endpoints,range_harm=harm)


def grouped(rows, fields):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    return [dict(zip(fields,key),**summarize_rows(values)) for key,values in sorted(groups.items())]


def run(source, previous, output):
    started=time.perf_counter()
    source,previous,output=source.resolve(),previous.resolve(),output.resolve()
    hashes=verify_archive(source);previous_hashes=verify_archive(previous)
    old_manifest=json.loads((previous/'manifest.json').read_text())
    if hashes!=old_manifest['source_hashes']:
        raise ValueError('previous audit used different source')
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    for name in ('experiments/audit_range_prediction_attribution.py','experiments/analyze_motion_time_structure.py',
                 'tests/test_range_prediction_attribution.py','docs/superpowers/plans/2026-10-05-range-prediction-attribution.md'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    manifest=dict(source=str(source),previous=str(previous),source_hashes=hashes,previous_hashes=previous_hashes,
        complete=False,truth_used=False,parameters_fitted=False,estimator_changed=False,
        scope='Frozen posthoc mean/covariance swaps, not independent or realizable estimator arms.')
    dump(output/'manifest.json',manifest)
    rows=[];solver_checks=[];calibration_checks=[];support_checks=[];temporal_coverage=[]
    for dataset in (1,4):
        events=[json.loads(line) for line in (source/f'd{dataset}/events.jsonl').read_text().splitlines()]
        warm=json.loads((source/f'd{dataset}/warm_start.json').read_text())
        first=min(p['timestamp_s'] for r in warm['robots'] for p in r['poses'])
        cutoff=max(p['timestamp_s'] for r in warm['robots'] for p in r['poses'])
        temporal_coverage.append(dict(dataset=dataset,heldout_seconds=first+120.-cutoff,
            forecast_window_seconds=15.,heldout_observations=sum(e['kind']=='observation' and
                cutoff<e['timestamp_s']<=first+120. for e in events)))
        fit=json.loads((source/f'd{dataset}/fits/joint_qr/fit.json').read_text())
        if fit['truth_used']:
            raise ValueError('truth-assisted calibration')
        eigen=fit['fisher_eigenvalues']
        calibration_checks.append(dict(dataset=dataset,scales=fit['scales'],success=fit['success'],
            component_rank=fit['component_rank'],residual_dof=fit['residual_dof'],
            at_numerical_boundary=fit['at_numerical_boundary'],max_log_score=float(np.max(np.abs(fit['log_score']))),
            fisher_eigenvalues=eigen,fisher_condition=max(eigen)/min(eigen),fixed_linearization=fit['fixed_linearization']))
        for elapsed in (30,60,90):
            forecasts=[]
            for arm in ('nominal','joint_qr'):
                folder=source/f'd{dataset}/t{elapsed}_{arm}'
                forecast=json.loads((folder/'forecast.json').read_text())
                summary=json.loads((folder/'summary.json').read_text())
                package=json.loads(folder.with_suffix('.json').read_text())
                if summary['truth_used']:
                    raise ValueError('truth-assisted state')
                for p in forecast['predictions']:
                    duration=p['timestamp_s']-forecast['checkpoint_s']
                    if not (0<duration<=forecast['horizon_s'] and abs(duration-p['horizon_s'])<1e-9):
                        raise ValueError('invalid forecast temporal support')
                forecasts.append(forecast)
                support_checks.append(dict(dataset=dataset,elapsed=elapsed,arm=arm,
                    **audit_forecast_support(forecast,package,events)))
                solver_checks.append(dict(dataset=dataset,elapsed=elapsed,arm=arm,
                    **{k:summary[k] for k in ('status','mode','scope','lm_iterations','posthoc_gradient_norm','final_cost')},
                    covariance_solve_relative_residual=forecast['covariance_solve_relative_residual'],
                    unknown_object_observations=forecast['unknown_object_observations']))
            rows.extend(dict(dataset=dataset,elapsed=elapsed,**paired_prediction(a,b))
                        for a,b in matched_predictions(*forecasts))
    groups={name:grouped(rows,fields) for name,fields in (
        ('dataset',('dataset',)),('robot',('dataset','robot')),
        ('checkpoint',('dataset','robot','elapsed')),('subject',('dataset','robot','subject')),
        ('checkpoint_subject',('dataset','robot','elapsed','subject')),
        ('frame',('dataset','robot','elapsed','timestamp_s')))}
    previous_groups=json.loads((previous/'score_components.json').read_text())['groups']
    largest=0.
    for group in groups['robot']:
        for arm,cell in (('nominal','m0_u0_r0'),('joint_qr','m1_u1_r1')):
            prior=next(g for g in previous_groups if (g['dataset'],g['robot'],g['arm'])==
                       (group['dataset'],group['robot'],arm))
            if group['observations']!=prior['count']:
                raise ValueError('previous observation support changed')
            largest=max(largest,max(abs(group['scores'][cell][k]-prior[k]) for k in SCORE_KEYS[:5]))
    if largest>1e-10:
        raise ValueError('previous group score changed')
    groups.update(solver_checks=solver_checks,calibration_checks=calibration_checks,
        support_checks=support_checks,temporal_coverage=temporal_coverage,
        warning='All summaries descriptive. Frame aggregation does not establish independent samples.')
    dump(output/'rows.json',dict(rows=rows,truth_used=False))
    dump(output/'summary.json',groups)
    manifest.update(complete=True,matched_observations=len(rows),hypothetical_scores=8*len(rows),
        max_score_parity=max(r['max_score_parity'] for r in rows),max_previous_group_parity=largest,
        max_observation_difference=max(r['max_observation_difference'] for r in rows),
        wall_seconds=time.perf_counter()-started)
    dump(output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('matched_observations','hypothetical_scores','max_score_parity',
        'max_previous_group_parity','max_observation_difference','wall_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','previous','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.previous,args.output)
