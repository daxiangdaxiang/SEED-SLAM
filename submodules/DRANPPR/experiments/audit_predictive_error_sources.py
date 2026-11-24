"""GT-free pair evidence followed by explicitly reference-assisted error diagnosis."""
import argparse
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.audit_native_checkpoint import sha256
from experiments.audit_lazy_refresh import dump
from experiments.analyze_frozen_noise_completion import moments
from experiments.evaluate_incremental_reference import read_rows
from experiments.network_noise_prediction import polar_model, propagate

COMPONENTS = ('measurement_reference', 'map', 'motion', 'checkpoint')


def difference(left, right):
    result = np.asarray(left)-np.asarray(right)
    result[0] = math.atan2(math.sin(result[0]), math.cos(result[0]))
    return result


def decompose(measured, estimated_pose, estimated_map, reference_pose, reference_map, reset_pose):
    ee = polar_model(estimated_pose, estimated_map)[0]
    re = polar_model(reference_pose, estimated_map)[0]
    er = polar_model(estimated_pose, reference_map)[0]
    rr = polar_model(reference_pose, reference_map)[0]
    ze = polar_model(reset_pose, estimated_map)[0]
    zr = polar_model(reset_pose, reference_map)[0]
    innovation = difference(measured, ee)
    terms = dict(measurement_reference=difference(measured,rr), map=difference(rr,re),
                 motion=difference(re,ze), checkpoint=difference(ze,ee))
    reverse = dict(measurement_reference=terms['measurement_reference'], map=difference(er,ee),
                   motion=difference(rr,zr), checkpoint=difference(zr,er))
    error = max(np.max(np.abs(difference(sum(values.values()), innovation))) for values in [terms,reverse])
    if error > 1e-10:
        raise ValueError('error decomposition failed to reconstruct innovation')
    return dict(innovation=innovation.tolist(), components={k:v.tolist() for k,v in terms.items()},
        reversed_components={k:v.tolist() for k,v in reverse.items()}, reconstruction_max_abs=float(error),
        counterfactuals={k:difference(measured,v).tolist() for k,v in
                        dict(reference_pose=re,reference_map=er,both_reference=rr,reference_checkpoint=ze).items()})


def summarize(rows):
    if not rows:
        return dict(count=0)
    def polar_stats(values):
        return {name:moments([v[i] for v in values]) for i,name in enumerate(['bearing','range'])}
    terms = np.array([[r['components'][k][1] for k in COMPONENTS] for r in rows])
    cross = terms.T@terms/len(rows)
    return dict(count=len(rows), innovation=polar_stats([r['innovation'] for r in rows]),
        components={k:polar_stats([r['components'][k] for r in rows]) for k in COMPONENTS},
        reversed_components={k:polar_stats([r['reversed_components'][k] for r in rows]) for k in COMPONENTS}
            if all('reversed_components' in r for r in rows) else {},
        counterfactuals={k:polar_stats([r['counterfactuals'][k] for r in rows]) for k in rows[0]['counterfactuals']},
        range_component_cross_moment=cross.tolist(), range_reconstructed_mse=float(cross.sum()),
        component_order=list(COMPONENTS))


def pair_measurement(left, right, scales):
    scales = np.asarray(scales,dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0.):
        raise ValueError('invalid pair noise')
    points = [r['range_m']*np.array([math.cos(r['bearing_rad']),math.sin(r['bearing_rad'])]) for r in [left,right]]
    delta = points[0]-points[1]
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError('degenerate pair distance')
    direction = delta/length
    variance = 0.
    for r in [left,right]:
        c, s = math.cos(r['bearing_rad']), math.sin(r['bearing_rad'])
        db = r['range_m']*float(direction@[-s,c])
        dr = float(direction@[c,s])
        variance += db**2*.05**2*scales[0]+dr**2*.1**2*scales[1]
    return length, float(variance)


def pair_evidence(events, support, scales, cutoff):
    observations, previous = [], -math.inf
    for e in events:
        t = e['timestamp_s']
        if t < previous or not math.isfinite(t):
            raise ValueError('invalid event order')
        previous = t
        if e['available_time_s'] != t or e['support_max_time_s'] > t:
            raise ValueError('invalid event support')
        if e['kind'] == 'observation':
            observations.append(e)
    past, rows, supported = {}, [], set()
    for (t,robot), group in itertools.groupby(observations,key=lambda e:(e['timestamp_s'],e['robot'])):
        frame = sorted(group,key=lambda e:e['landmark_subject'])
        if len({e['landmark_subject'] for e in frame}) != len(frame):
            raise ValueError('duplicate subject in a frame')
        for a,b in itertools.combinations(frame,2):
            key = (robot,a['landmark_subject'],b['landmark_subject'])
            value, variance = pair_measurement(a,b,[1.,1.] if t <= cutoff else scales)
            if key in past:
                old, old_variance, old_time = past[key]
                if old_time >= t:
                    raise ValueError('nonconsecutive pair frame order')
                if all((robot,e['source_row']) in support for e in [a,b]):
                    rows.append(dict(robot=robot,timestamp_s=t,previous_timestamp_s=old_time,
                        subject_a=key[1],subject_b=key[2],source_row_a=a['source_row'],source_row_b=b['source_row'],
                        difference_m=value-old,variance_m2=variance+old_variance,
                        normalized_squared=(value-old)**2/(variance+old_variance)))
                    supported.update((robot,e['source_row']) for e in [a,b])
            past[key] = value,variance,t
    return dict(pairs=rows,supported_observations=len(supported),requested_observations=len(support),
                truth_used=False,state_used=False)


def audit_window(package, states, predicted, events, alignment, truth, labels):
    shift, angle = np.array(alignment[:2]), alignment[2]
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c,-s],[s,c]])
    pose_rows = {(int(r['robot']),int(r['index'])):np.array([float(r[k]) for k in ['x','y','yaw']])
                 for r in states if r['kind']=='pose'}
    points = {int(r['cluster']):rotation@np.array([float(r[k]) for k in ['x','y']])+shift
              for r in states if r['kind']=='landmark'}
    last, known = {}, {}
    for robot in package['robots']:
        r = robot['id']
        final = max(robot['poses'],key=lambda p:p['timestamp_s'])
        estimate = pose_rows[r,final['index']]
        xy,yaw = truth.robots[r].interpolate(final['timestamp_s'])
        last[r] = dict(estimate=np.r_[rotation@estimate[:2]+shift,estimate[2]+angle],
                       reset=np.r_[xy,yaw],index=final['index'],time=final['timestamp_s'])
        for obj in robot['objects']:
            known[r,obj['label']] = obj['cluster']
    expected = {(r['robot'],r['source_row']):r for r in predicted['predictions']}
    if len(expected) != len(predicted['predictions']):
        raise ValueError('duplicate forecast identity')
    output = []
    for e in events:
        t = e['timestamp_s']
        if t <= predicted['checkpoint_s']:
            continue
        if t > predicted['checkpoint_s']+predicted['horizon_s']:
            break
        current = last[e['robot']]
        if e['kind']=='pose':
            if e['from_pose_index'] != current['index'] or e['pose_index'] != current['index']+1:
                raise ValueError('nonconsecutive propagation')
            for mode in ['estimate','reset']:
                current[mode] = propagate(current[mode],np.eye(3),np.zeros((3,3)),e['delta'],e['duration_s'],np.ones(3))[0]
            current.update(index=e['pose_index'],time=t)
            continue
        key = (e['robot'],e['source_row'])
        if key not in expected:
            continue
        old = expected.pop(key)
        if e['pose_index'] != current['index'] or t != current['time'] or t != old['timestamp_s']:
            raise ValueError('forecast frame mismatch')
        point = points[known[e['robot'],e['landmark_subject']]]
        measured = [e['bearing_rad'],e['range_m']]
        mean = polar_model(current['estimate'],point)[0]
        parity = max(np.max(np.abs(difference(mean,old['mean']))),
                     np.max(np.abs(difference(difference(measured,mean),old['innovation']))))
        if parity > 1e-9:
            raise ValueError('archived forecast parity failed')
        xy,yaw = truth.robots[e['robot']].interpolate(t)
        reference_map = truth.landmarks[labels.get(e['landmark_subject'],e['landmark_subject'])]
        terms = decompose(measured,current['estimate'],point,np.r_[xy,yaw],reference_map,current['reset'])
        output.append(dict(robot=e['robot'],source_row=e['source_row'],timestamp_s=t,subject=e['landmark_subject'],
            horizon_s=t-predicted['checkpoint_s'],forecast_parity_max_abs=float(parity),**terms))
    if expected:
        raise ValueError('missing forecast support')
    return output


def stratify(rows, pair_results):
    subjects, pairs = [], []
    identities = sorted({(r['dataset'],r['arm'],r['robot'],r['subject']) for r in rows})
    for identity in identities:
        selected = [r['components']['measurement_reference'] for r in rows
                    if (r['dataset'],r['arm'],r['robot'],r['subject'])==identity]
        subjects.append(dict(zip(['dataset','arm','robot','subject'],identity),
            bearing=moments([v[0] for v in selected]),range=moments([v[1] for v in selected])))
    for group in pair_results:
        for robot in range(1,6):
            selected = [r for r in group['pairs'] if r['robot']==robot]
            row = dict(dataset=group['dataset'],arm=group['arm'],robot=robot,pair_count=len(selected))
            if selected:
                energy = np.array([r['normalized_squared'] for r in selected])
                row.update(mean_energy=float(energy.mean()),
                    energy_quantiles=np.quantile(energy,[.5,.9,.95,.99]).tolist(),
                    unique_current_observations=len({r[k] for r in selected for k in ['source_row_a','source_row_b']}),
                    difference=moments([r['difference_m'] for r in selected]),
                    top_ten_percent_energy_fraction=float(np.sort(energy)[-math.ceil(.1*len(energy)):].sum()/energy.sum())
                        if energy.sum() else 0.)
            pairs.append(row)
    return dict(reference_by_subject=subjects,pairs=pairs,post_selected_stratification=True,
        caveat='Descriptive only. Pair statistics share observations and are not independent trials; a large difference does not identify which endpoint is wrong.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source,output = args.source.resolve(),args.output.resolve()
    prior = json.loads((source/'manifest.json').read_text())
    if not prior['all_estimators_terminal'] or not prior['all_predictions_terminal']:
        raise ValueError('source producer unfinished')
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,name = line.split('  ',1)
        if sha256(source/name) != digest:
            raise ValueError('source archive hash mismatch: '+name)
        hashes[name] = digest
    output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = ['experiments/audit_predictive_error_sources.py','experiments/network_noise_prediction.py',
        'experiments/analyze_frozen_noise_completion.py','experiments/evaluate_incremental_reference.py',
        'benchmarks/mrclam/evaluate.py','benchmarks/mrclam/source.py','tests/test_predictive_error_sources.py',
        'docs/superpowers/plans/2026-10-04-predictive-error-sources.md']
    for name in files:
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    manifest = dict(source=str(source),source_hashes=hashes,estimator_changed=False,parameters_fitted=False,
        gt_free_stage_finished=False,reference_stage_finished=False,post_selected_cases=True,
        protocol='Frozen forecast support; raw pair evidence first, reference-assisted diagnosis second; never estimator input')
    dump(output/'manifest.json',manifest)
    analysis = json.loads((source/'analysis.json').read_text())
    producer = json.loads((source/'producer.json').read_text())
    streams,window_data,pair_results = {},{},[]
    for dataset in (1,4):
        warm = json.loads((source/f'd{dataset}/warm_start.json').read_text())
        cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
        first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
        with (source/f'd{dataset}/events.jsonl').open() as stream:
            streams[dataset] = []
            for line in stream:
                e = json.loads(line)
                if e['timestamp_s'] > first+95.:
                    break
                streams[dataset].append(e)
        for arm in ('nominal','joint_qr'):
            support = set()
            for elapsed in (30,60,90):
                folder = source/f'd{dataset}/t{elapsed}_{arm}'
                predicted = json.loads((folder/'forecast.json').read_text())
                window_data[dataset,arm,elapsed] = predicted
                support.update((r['robot'],r['source_row']) for r in predicted['predictions'])
            scales = producer['fits'][str(dataset)]['arms'][arm]['scales'][3:]
            result = pair_evidence(streams[dataset],support,scales,cutoff)
            result.update(dataset=dataset,arm=arm)
            pair_results.append(result)
    dump(output/'pose_free_pairs.json',dict(results=pair_results,truth_used=False,state_used=False,
        caveat='First-order measurement variance; repeated pairs and same-frame pairs are dependent. No independent p-values or outlier labels.'))
    manifest['gt_free_stage_finished'] = True
    dump(output/'manifest.json',manifest)

    # Reference loading is deliberately below the saved GT-free evidence stage.
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    rows,groups = [],[]
    for dataset in (1,4):
        data = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        raw = audit_dataset(data,args.raw)
        truth = load_ground_truth(data,raw.dataset_directory)
        labels = dict(data.evaluation.object_ground_truth_label_map)
        for arm in ('nominal','joint_qr'):
            for elapsed in (30,60,90):
                folder = source/f'd{dataset}/t{elapsed}_{arm}'
                quality = next(r for r in analysis['results'] if (r['dataset'],r['arm'],r['elapsed'])==(dataset,arm,elapsed))
                alignment = [quality[k] for k in ['alignment_x','alignment_y','alignment_yaw']]
                records = audit_window(json.loads(folder.with_suffix('.json').read_text()),
                    read_rows(folder/'final_estimates.csv'),window_data[dataset,arm,elapsed],
                    streams[dataset],alignment,truth,labels)
                rows.extend(dict(dataset=dataset,arm=arm,elapsed=elapsed,**r) for r in records)
            for robot in range(1,6):
                for elapsed in [None,30,60,90]:
                    selected = [r for r in rows if (r['dataset'],r['arm'],r['robot'])==(dataset,arm,robot)
                                and (elapsed is None or r['elapsed']==elapsed)]
                    groups.append(dict(dataset=dataset,arm=arm,robot=robot,elapsed=elapsed,**summarize(selected)))
    dump(output/'reference_components.json',dict(rows=rows,truth_used=True,estimator_feedback=False))
    dump(output/'stratified_diagnostics.json',stratify(rows,pair_results))
    dump(output/'summary.json',dict(groups=groups,max_forecast_parity=max(r['forecast_parity_max_abs'] for r in rows),
        max_reconstruction_error=max(r['reconstruction_max_abs'] for r in rows),
        warning='Reference-assisted, ordering-dependent attribution. Reference error is not pure sensor noise. Variance cross terms must not be dropped.'))
    manifest.update(reference_stage_finished=True,reference_used_only_posthoc=True,wall_seconds=time.perf_counter()-started)
    dump(output/'manifest.json',manifest)
    for group in groups:
        if group['arm']=='joint_qr' and group['elapsed'] is None:
            print(json.dumps(group),flush=True)


if __name__ == '__main__':
    main()
