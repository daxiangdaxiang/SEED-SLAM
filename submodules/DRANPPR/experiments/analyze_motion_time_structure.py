"""Posthoc motion summaries and exact, GT-free forecast score decomposition."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.audit_lazy_refresh import dump
from experiments.audit_motion_time_structure import verify_archive


def split_score(innovation, covariance):
    v,c=np.asarray(innovation,float),np.asarray(covariance,float)
    if (v.shape!=(2,) or c.shape!=(2,2) or not np.isfinite(v).all() or
            not np.isfinite(c).all() or not np.allclose(c,c.T,rtol=1e-12,atol=1e-14)):
        raise ValueError('invalid Gaussian prediction')
    try:
        chol=np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError('nonpositive predictive covariance') from error
    whitened=np.linalg.solve(chol,v)
    joint=float(.5*(whitened@whitened+2*np.log(np.diag(chol)).sum()+2*np.log(2*np.pi)))
    result=dict(joint=joint)
    for i,name,conditional in ((0,'bearing','range_given_bearing'),(1,'range','bearing_given_range')):
        j=1-i
        variance=c[j,j]-c[j,i]*c[i,j]/c[i,i]
        error=v[j]-c[j,i]/c[i,i]*v[i]
        result[name]=float(.5*(v[i]**2/c[i,i]+np.log(2*np.pi*c[i,i])))
        result[conditional]=float(.5*(error**2/variance+np.log(2*np.pi*variance)))
        if not math.isclose(joint,result[name]+result[conditional],rel_tol=1e-11,abs_tol=1e-11):
            raise ValueError('Gaussian chain rule mismatch')
    return result


def analyze(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    if not manifest['reference_stage_finished']:
        raise ValueError('incomplete motion audit')
    source=Path(manifest['source'])
    if verify_archive(source)!=manifest['source_hashes']:
        raise ValueError('forecast archive changed')
    scores=[];groups=[];max_error=0.;keys={}
    for dataset in (1,4):
        for arm in ('nominal','joint_qr'):
            keys[dataset,arm]=set()
            for elapsed in (30,60,90):
                forecast=json.loads((source/f'd{dataset}/t{elapsed}_{arm}/forecast.json').read_text())
                if forecast['truth_used']:
                    raise ValueError('reference-tainted forecast')
                for p in forecast['predictions']:
                    key=(elapsed,p['robot'],p['source_row'],p['timestamp_s'])
                    if key in keys[dataset,arm]:
                        raise ValueError('duplicate forecast row')
                    keys[dataset,arm].add(key)
                    split=split_score(p['innovation'],p['covariance'])
                    error=abs(split['joint']-p['nll']);max_error=max(max_error,error)
                    if error>1e-10:
                        raise ValueError('archived score mismatch')
                    scores.append(dict(dataset=dataset,arm=arm,elapsed=elapsed,
                        robot=p['robot'],source_row=p['source_row'],
                        innovation=p['innovation'],covariance=p['covariance'],**split))
            for robot in range(1,6):
                rows=[r for r in scores if (r['dataset'],r['arm'],r['robot'])==(dataset,arm,robot)]
                groups.append(dict(dataset=dataset,arm=arm,robot=robot,count=len(rows),
                    bearing_rmse_rad=float(np.sqrt(np.mean([r['innovation'][0]**2 for r in rows]))),
                    range_rmse_m=float(np.sqrt(np.mean([r['innovation'][1]**2 for r in rows]))),
                    mean_bearing_variance=float(np.mean([r['covariance'][0][0] for r in rows])),
                    mean_range_variance=float(np.mean([r['covariance'][1][1] for r in rows])),
                    **{k:float(np.mean([r[k] for r in rows])) for k in
                       ('joint','bearing','range_given_bearing','range','bearing_given_range')}))
        if keys[dataset,'nominal']!=keys[dataset,'joint_qr']:
            raise ValueError('unequal forecast support')
    bearing=json.loads((output/'bearing_attribution.json').read_text())['rows']
    uniform=json.loads((output/'uniform_increments.json').read_text())['rows']
    motions=[];blocks=[]
    for dataset in (1,4):
        for robot in range(1,6):
            rows=[r for r in bearing if (r['dataset'],r['robot'])==(dataset,robot)]
            summary=dict(dataset=dataset,robot=robot,observations=len(rows),
                distinct_frames=len({(r['elapsed'],r['timestamp_s']) for r in rows}))
            for k in ('motion_bearing_rad','heading_part_rad','position_part_rad'):
                values=np.array([r[k] for r in rows])
                summary[k]=dict(mean=float(values.mean()),rmse=float(np.sqrt(np.mean(values**2))))
            motions.append(summary)
            errors=np.array([r['error_rad'] for r in uniform if (r['dataset'],r['robot'])==(dataset,robot)])
            complete=len(errors)//10
            values=errors[:10*complete].reshape(complete,10).sum(axis=1)
            blocks.append(dict(dataset=dataset,robot=robot,seconds=10,errors_rad=values.tolist(),
                unused_seconds=len(errors)%10,positive=int(sum(values>0)),negative=int(sum(values<0))))
    dump(output/'score_components.json',dict(rows=scores,groups=groups,max_score_parity=max_error,
        truth_used=False,estimator_feedback=False,
        scope='Posthoc Gaussian chain-rule diagnostics, both orders; not causal attribution.'))
    dump(output/'motion_components.json',dict(groups=motions,ten_second_blocks=blocks,truth_used=True,
        scope='Reference-assisted descriptive summaries; repeated observations are not independent.'))
    print(json.dumps(dict(score_rows=len(scores),score_groups=len(groups),max_score_parity=max_error)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
