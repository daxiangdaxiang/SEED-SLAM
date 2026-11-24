"""Reference-assisted motion diagnostics; no noise fit or estimator feedback."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from benchmarks.mrclam.extract import (
    PlanarPose, OdometrySample, _merge_duplicate_odometry,
    exact_unicycle_increment, wrap_angle,
)
from benchmarks.mrclam.source import read_numeric_table
from experiments.audit_native_checkpoint import sha256
from experiments.audit_lazy_refresh import dump
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_prediction import propagate, polar_model
from experiments.audit_predictive_error_sources import difference


def canonical_control_table(table):
    merged=_merge_duplicate_odometry([OdometrySample(*row) for row in table])
    return np.array([[r.timestamp_s,r.forward_velocity_mps,r.angular_velocity_radps] for r in merged])


class HeldCommands:
    def __init__(self, samples):
        self.samples = np.asarray(samples,dtype=float)
        if (self.samples.ndim!=2 or self.samples.shape[1]!=3 or len(self.samples)<2 or
                not np.isfinite(self.samples).all() or np.any(np.diff(self.samples[:,0])<0)):
            raise ValueError('invalid chronological controls')

    def increment(self, start, end):
        table=self.samples
        if not math.isfinite(start+end) or end<=start or start<table[0,0] or end>table[-1,0]:
            raise ValueError('interval outside control support')
        k=int(np.searchsorted(table[:,0],start,side='right'))-1
        current,pose,yaw,travel=start,PlanarPose.identity(),0.,0.
        while current<end:
            stop=min(end,table[k+1,0] if k+1<len(table) else end)
            dt=stop-current
            pose=pose.compose(exact_unicycle_increment(table[k,1],table[k,2],dt))
            yaw+=table[k,2]*dt
            travel+=table[k,1]*dt
            current=stop
            if k+1<len(table) and table[k+1,0]<=current:
                k=int(np.searchsorted(table[:,0],current,side='right'))-1
        return dict(delta=[pose.x,pose.y,pose.yaw],yaw_integral_rad=float(yaw),
            forward_integral_m=float(travel),last_control_time_s=float(table[k,0]))


def increment_statistics(errors, durations, density):
    e,dt=np.asarray(errors,float),np.asarray(durations,float)
    if (e.ndim!=1 or not len(e) or e.shape!=dt.shape or
            not np.isfinite(e).all() or not np.isfinite(dt).all() or np.any(dt<=0) or
            not math.isfinite(density) or density<=0):
        raise ValueError('invalid motion increments')
    rate=float(e.sum()/dt.sum())
    total=float(np.sum(e**2/dt)/density)
    coherent=float(e.sum()**2/(density*dt.sum()))
    scatter=float(np.sum((e-rate*dt)**2/dt)/density)
    if not math.isclose(total,coherent+scatter,rel_tol=1e-11,abs_tol=1e-11):
        raise ValueError('motion energy decomposition failed')
    return dict(count=len(e),total_seconds=float(dt.sum()),rate_radps=rate,
        total_energy=total,mean_energy=total/len(e),rate_energy=coherent,
        scatter_energy=scatter,scatter_energy_per_dof=scatter/(len(e)-1) if len(e)>1 else None,
        rate_energy_fraction=coherent/total if total else None,
        rmse_rad=float(np.sqrt(np.mean(e**2))))


def lag_correlation(errors, lag):
    e=np.asarray(errors,float)
    if lag<1 or not np.isfinite(e).all():
        raise ValueError('invalid lag series')
    if len(e)-lag<2 or np.std(e[:-lag])<1e-15 or np.std(e[lag:])<1e-15:
        return None
    return float(np.corrcoef(e[:-lag],e[lag:])[0,1])


def horizon_statistics(errors, density):
    errors=np.asarray(errors,float)
    output=[]
    for seconds in (1,2,5,10):
        count=len(errors)//seconds
        if not count:
            output.append(dict(seconds=seconds,blocks=0,unused_seconds=len(errors)))
            continue
        values=errors[:count*seconds].reshape(count,seconds).sum(axis=1)
        output.append(dict(seconds=seconds,blocks=count,unused_seconds=len(errors)%seconds,
            **increment_statistics(values,np.full(count,seconds),density)))
    return output


def reference_pose(series, timestamp):
    xy,yaw=series.interpolate(timestamp)
    return np.r_[xy,yaw]


def window_motion(package, states, events, truth, alignment, checkpoint, horizon, scales):
    angle=alignment[2]
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    poses={(int(r['robot']),int(r['index'])):np.array([float(r[k]) for k in ('x','y','yaw')])
           for r in states if r['kind']=='pose'}
    last={}
    for robot in package['robots']:
        rid=robot['id']; p=max(robot['poses'],key=lambda p:p['timestamp_s'])
        estimated=poses[rid,p['index']]
        estimated=np.r_[rotation@estimated[:2]+alignment[:2],estimated[2]+angle]
        ref=reference_pose(truth.robots[rid],p['timestamp_s'])
        last[rid]=dict(estimated=estimated,reset=ref,index=p['index'],time=p['timestamp_s'],
            initial_time=p['timestamp_s'],initial_error=wrap_angle(estimated[2]-ref[2]),
            covariance=np.zeros((3,3)),cumulative_increment=0.)
    rows=[]
    previous=checkpoint
    for e in events:
        t=e['timestamp_s']
        if t<=checkpoint:
            continue
        if t>checkpoint+horizon:
            break
        if t<previous:
            raise ValueError('nonchronological forecast')
        previous=t
        if e['kind']!='pose':
            continue
        if e['available_time_s']!=t or e['support_max_time_s']>t:
            raise ValueError('future control support')
        rid=e['robot'];current=last[rid]
        if e['from_pose_index']!=current['index'] or e['pose_index']!=current['index']+1:
            raise ValueError('nonconsecutive forecast poses')
        if abs(t-current['time']-e['duration_s'])>1e-9:
            raise ValueError('incorrect propagation duration')
        old_ref=reference_pose(truth.robots[rid],current['time'])
        ref=reference_pose(truth.robots[rid],t)
        error=wrap_angle(e['delta'][2]-wrap_angle(ref[2]-old_ref[2]))
        current['cumulative_increment']+=error
        current['reset'],_,current['covariance']=propagate(current['reset'],np.eye(3),
            current['covariance'],e['delta'],e['duration_s'],np.array(scales))
        current['estimated']=propagate(current['estimated'],np.eye(3),np.zeros((3,3)),
            e['delta'],e['duration_s'],np.array(scales))[0]
        motion=wrap_angle(current['reset'][2]-ref[2])
        total=wrap_angle(current['estimated'][2]-ref[2])
        parity=max(abs(wrap_angle(motion-current['cumulative_increment'])),
            abs(wrap_angle(total-motion-current['initial_error'])),
            abs(current['covariance'][2,2]-.05**2*scales[2]*(t-current['initial_time'])))
        if parity>1e-10:
            raise ValueError('heading decomposition or process variance failed')
        rows.append(dict(robot=rid,pose_index=e['pose_index'],timestamp_s=t,
            initial_timestamp_s=current['initial_time'],state_age_s=checkpoint-current['initial_time'],
            forecast_horizon_s=t-checkpoint,propagation_s=t-current['initial_time'],
            duration_s=e['duration_s'],yaw_increment_error_rad=error,
            motion_yaw_error_rad=motion,initial_yaw_error_rad=current['initial_error'],
            total_yaw_error_rad=total,process_yaw_variance=float(current['covariance'][2,2]),
            normalized_motion_yaw_energy=motion**2/current['covariance'][2,2],
            motion_position_error_m=float(np.linalg.norm(current['reset'][:2]-ref[:2])),
            reference_pose=ref.tolist(),reset_pose=current['reset'].tolist(),parity_error=parity))
        current.update(index=e['pose_index'],time=t)
    return rows


def verify_archive(path):
    hashes={}
    for line in (path/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha256(path/name)!=digest:
            raise ValueError('archive hash mismatch: '+name)
        hashes[name]=digest
    return hashes


def run(source, attribution, raw, output):
    started=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    source,attribution,output=source.resolve(),attribution.resolve(),output.resolve()
    hashes=verify_archive(source);previous_hashes=verify_archive(attribution)
    output.mkdir(parents=True,exist_ok=False)
    names=['experiments/audit_motion_time_structure.py','experiments/network_noise_prediction.py',
        'experiments/audit_predictive_error_sources.py','benchmarks/mrclam/extract.py',
        'benchmarks/mrclam/evaluate.py','benchmarks/mrclam/source.py',
        'tests/test_motion_time_structure.py','docs/superpowers/plans/2026-10-04-motion-time-structure.md']
    for name in names:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    manifest=dict(source=str(source),attribution=str(attribution),source_hashes=hashes,
        attribution_hashes=previous_hashes,gt_free_stage_finished=False,
        reference_stage_finished=False,parameters_fitted=False,estimator_changed=False,
        raw_hashes={},posthoc_reference_only=True)
    dump(output/'manifest.json',manifest)
    datasets={};integration=[]
    for dataset in (1,4):
        warm=json.loads((source/f'd{dataset}/warm_start.json').read_text())
        first=min(p['timestamp_s'] for r in warm['robots'] for p in r['poses'])
        cutoff=max(p['timestamp_s'] for r in warm['robots'] for p in r['poses'])
        folder=raw/{1:'MRCLAM_Dataset1',4:'MRSLAM_Dataset4'}[dataset]
        metadata=json.loads((source/f'd{dataset}/metadata.json').read_text())
        commands={}
        for robot in range(1,6):
            name=f'Robot{robot}_Odometry.dat'; path=folder/name
            digest=sha256(path)
            if digest!=metadata['source_hashes'][name]:
                raise ValueError('raw command hash mismatch')
            manifest['raw_hashes'][str(path)]=digest
            commands[robot]=HeldCommands(canonical_control_table(read_numeric_table(path,3,name)))
        events=[]
        with (source/f'd{dataset}/events.jsonl').open() as stream:
            for line in stream:
                e=json.loads(line)
                if e['timestamp_s']>first+120.:
                    break
                events.append(e)
        last={};largest=0.;count=0
        for e in events:
            if e['kind']!='pose':continue
            rid=e['robot'];t=e['timestamp_s']
            if e['pose_index']:
                delta=commands[rid].increment(last[rid],t)['delta']
                delta_error=np.array(delta)-e['delta'];delta_error[2]=wrap_angle(delta_error[2])
                error=float(np.max(np.abs(delta_error)))
                largest=max(largest,error);count+=1
                if error>1e-9:raise ValueError('raw control/stream parity failed')
            last[rid]=t
        integration.append(dict(dataset=dataset,increments=count,max_error=largest,truth_used=False))
        datasets[dataset]=dict(warm=warm,first=first,cutoff=cutoff,folder=folder,commands=commands,events=events)
    dump(output/'integration_parity.json',integration)
    manifest['gt_free_stage_finished']=True
    dump(output/'manifest.json',manifest)

    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset,load_manifest
    quality=json.loads((source/'analysis.json').read_text())['results']
    prior_rows=json.loads((attribution/'reference_components.json').read_text())['rows']
    prior={(r['dataset'],r['elapsed'],r['robot'],r['source_row']):r
           for r in prior_rows if r['arm']=='joint_qr'}
    frame_rows=[];observation_rows=[];uniform_rows=[];window_groups=[];uniform_groups=[]
    for dataset,data in datasets.items():
        declared=load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audited=audit_dataset(declared,raw)
        truth=load_ground_truth(declared,audited.dataset_directory)
        for robot in range(1,6):
            path=data['folder']/f'Robot{robot}_Groundtruth.dat'
            manifest['raw_hashes'][str(path)]=sha256(path)
        scales=json.loads((source/f'd{dataset}/fits/joint_qr/fit.json').read_text())['scales']
        for elapsed in (30,60,90):
            folder=source/f'd{dataset}/t{elapsed}_joint_qr'
            package=json.loads(folder.with_suffix('.json').read_text())
            states=read_rows(folder/'final_estimates.csv')
            q=next(r for r in quality if (r['dataset'],r['arm'],r['elapsed'])==(dataset,'joint_qr',elapsed))
            alignment=[q[k] for k in ('alignment_x','alignment_y','alignment_yaw')]
            rows=window_motion(package,states,data['events'],truth,alignment,data['first']+elapsed,5.,scales[:3])
            index={(r['robot'],r['timestamp_s']):r for r in rows}
            if len(index)!=len(rows):raise ValueError('duplicate motion frames')
            frame_rows.extend(dict(dataset=dataset,elapsed=elapsed,**r) for r in rows)
            angle=alignment[2];rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            points={int(r['cluster']):rotation@np.array([float(r['x']),float(r['y'])])+alignment[:2]
                    for r in states if r['kind']=='landmark'}
            labels={(r['id'],o['label']):o['cluster'] for r in package['robots'] for o in r['objects']}
            predicted=json.loads((folder/'forecast.json').read_text())['predictions']
            for p in predicted:
                frame=index[p['robot'],p['timestamp_s']]
                point=points[labels[p['robot'],p['subject']]]
                motion=difference(polar_model(frame['reference_pose'],point)[0],polar_model(frame['reset_pose'],point)[0])
                old=prior[dataset,elapsed,p['robot'],p['source_row']]['components']['motion']
                parity=float(np.max(np.abs(difference(motion,old))))
                if parity>1e-10:raise ValueError('previous motion attribution changed')
                observation_rows.append(dict(dataset=dataset,elapsed=elapsed,robot=p['robot'],
                    source_row=p['source_row'],timestamp_s=p['timestamp_s'],
                    motion_bearing_rad=float(motion[0]),heading_part_rad=frame['motion_yaw_error_rad'],
                    position_part_rad=wrap_angle(motion[0]-frame['motion_yaw_error_rad']),parity_error=parity))
            for robot in range(1,6):
                selected=[r for r in rows if r['robot']==robot]
                group=dict(dataset=dataset,elapsed=elapsed,robot=robot,frames=len(selected),
                    observations=sum(p['robot']==robot for p in predicted))
                if selected:
                    group.update(increments=increment_statistics([r['yaw_increment_error_rad'] for r in selected],
                        [r['duration_s'] for r in selected],.05**2*scales[2]),
                        motion_yaw_rmse_rad=float(np.sqrt(np.mean([r['motion_yaw_error_rad']**2 for r in selected]))),
                        mean_cumulative_energy=float(np.mean([r['normalized_motion_yaw_energy'] for r in selected])),
                        max_cumulative_energy=max(r['normalized_motion_yaw_energy'] for r in selected),
                        initial_yaw_error_rad=selected[0]['initial_yaw_error_rad'],
                        state_age_s=selected[0]['state_age_s'])
                window_groups.append(group)
        start=data['first']+math.ceil(data['cutoff']-data['first'])
        count=int(round(data['first']+120.-start))
        for robot in range(1,6):
            series=truth.robots[robot];errors=[]
            for i in range(count):
                t=start+i
                command=data['commands'][robot].increment(t,t+1.)
                angle=np.interp([t,t+1.],series.timestamps_s,series.unwrapped_yaws_rad)
                error=float(command['yaw_integral_rad']-(angle[1]-angle[0]))
                errors.append(error)
                uniform_rows.append(dict(dataset=dataset,robot=robot,start_s=t,end_s=t+1.,
                    error_rad=error,command_yaw_rad=command['yaw_integral_rad'],
                    reference_yaw_rad=float(angle[1]-angle[0]),forward_command_m=command['forward_integral_m']))
            uniform_groups.append(dict(dataset=dataset,robot=robot,start_elapsed_s=start-data['first'],
                end_elapsed_s=120.,seconds=count,q_yaw=scales[2],
                lag_correlations={str(lag):lag_correlation(errors,lag) for lag in (1,2,5)},
                horizons=horizon_statistics(errors,.05**2*scales[2])))
    dump(output/'motion_frames.json',dict(rows=frame_rows,truth_used=True,estimator_feedback=False))
    dump(output/'bearing_attribution.json',dict(rows=observation_rows,truth_used=True))
    dump(output/'uniform_increments.json',dict(rows=uniform_rows,truth_used=True))
    dump(output/'summary.json',dict(windows=window_groups,uniform=uniform_groups,
        max_motion_parity=max(r['parity_error'] for r in frame_rows),
        max_bearing_parity=max(r['parity_error'] for r in observation_rows),
        scope='Descriptive reference-assisted audit. No independent p-values, bias correction, or sensor-noise identification.'))
    manifest.update(reference_stage_finished=True,wall_seconds=time.perf_counter()-started)
    dump(output/'manifest.json',manifest)
    print(json.dumps(dict(output=str(output),frame_rows=len(frame_rows),observation_rows=len(observation_rows),
        uniform_increments=len(uniform_rows),wall_seconds=manifest['wall_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','attribution','raw','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.attribution,args.raw,args.output)


if __name__=='__main__':
    main()
