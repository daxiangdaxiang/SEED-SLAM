"""Read-only audit of nonlinear chart invalidation; not a runtime benchmark."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.audit_conditional_replay import compare_runs, read_rows, sha256
from experiments.audit_native_joint_replay import validate_ledger


def summarize_records(records):
    result=dict(records=0,stages={},robots={})
    origins=[]
    additive=['pose_count','point_count','requested_pose_keys','requested_point_keys',
        'raw_trigger_pose_keys','raw_trigger_point_keys','aligned_trigger_pose_keys',
        'aligned_trigger_point_keys','pose_translation_raw_sq','pose_translation_aligned_sq',
        'factors','touched_factors','pose_only_factors','point_only_factors','both_factors']
    maxima=['max_raw_delta','max_aligned_delta','fixed_aligned_max']
    def merge(group,record):
        group['records']=group.get('records',0)+1
        for field in additive:
            group[field]=group.get(field,0)+record[field]
        for field in maxima:
            group[field]=max(group.get(field,0),record[field])
        families=group.setdefault('families',{})
        for name,values in record['families'].items():
            if name not in families:
                families[name]=copy.deepcopy(values)
                continue
            for field,value in values.items():
                if field.endswith('_bins'):
                    families[name][field]=[a+b for a,b in zip(families[name][field],value)]
                elif field.endswith('_max'):
                    families[name][field]=max(families[name][field],value)
                else:
                    families[name][field]+=value
    for record in records:
        if 'pose_origin' in record:
            origins.append(dict(record['pose_origin'],robot=record['robot'],stage=record['stage'],epoch=record['epoch']))
        bins=record['norm_bin_upper_bounds']
        if result.setdefault('norm_bin_upper_bounds',bins)!=bins:
            raise ValueError('incompatible diagnostic bins')
        result['records']+=1
        merge(result['stages'].setdefault(record['stage'],{}),record)
        merge(result['robots'].setdefault(str(record['robot']),{}).setdefault(record['stage'],{}),record)
    for group in list(result['stages'].values())+[g for r in result['robots'].values() for g in r.values()]:
        energy=group['pose_translation_raw_sq']
        group['pose_translation_removed_fraction']=1.-group['pose_translation_aligned_sq']/energy if energy else None
        for family in group['families'].values():
            dimensions=family['residual_dimensions']
            for field in ['raw_error','transported_error','residual_change']:
                family[field+'_rms_per_dimension']=math.sqrt(family[field+'_sq']/dimensions) if dimensions else None
    if origins:
        result['pose_origin']=summarize_records(origins)
    return result


def verify_refresh_counts(folder,records):
    updates=read_rows(folder/'updates.csv')
    for epoch,row in enumerate(updates):
        local=[r for r in records if r['epoch']==epoch and r['stage']=='optimizer_refresh']
        if sum(r['touched_factors'] for r in local)!=int(row['factors_linearized']):
            raise ValueError('optimizer factor audit mismatch')
        if sum(r['requested_pose_keys']+r['requested_point_keys'] for r in local)!=int(row['refreshed_keys']):
            raise ValueError('optimizer variable audit mismatch')
    for r in records:
        if not 0<=r['epoch']<len(updates):
            raise ValueError('invalid epoch')
        if r['touched_factors']!=sum(r[k] for k in ['pose_only_factors','point_only_factors','both_factors']):
            raise ValueError('factor incidence mismatch')
        for family in r['families'].values():
            if family['gauge_invariance_max']>1e-8 or family['cached_residual_max']>1e-8:
                raise ValueError('inconsistent cached model or gauge transport')
            for field in ['raw_error_bins','transported_error_bins','residual_change_bins']:
                if sum(family[field])!=family['count']:
                    raise ValueError('histogram count mismatch')
    if records and 'pose_origin' in records[0]:
        verify_refresh_counts(folder,[dict(r['pose_origin'],robot=r['robot'],stage=r['stage'],epoch=r['epoch']) for r in records])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[1]
    binary=output/'native-joint-replay'; shutil.copy2(args.binary,binary)
    sources=[Path(__file__),source_root/'CMakeLists.txt',source_root/'cmake/native_local_model.cmake',
        source_root/'experiments/online_stream.h',source_root/'experiments/audit_conditional_replay.py',
        source_root/'experiments/audit_native_joint_replay.py']
    sources+=list((source_root/'experiments/native_local').glob('*'))
    for name in ['IncrementalLocalModel','IncrementalConditionalModel','NativeSchurModel','NativeLocalResponse',
                 'NativeInterface','NativeLocalOptimizer','NativeBetweenPose2Factor']:
        sources+=[source_root/f'include/DPGO/{name}.h',source_root/f'src/{name}.cpp']
    for path in sources:
        if path.is_file():
            target=output/'source'/path.relative_to(source_root)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    inputs=output/'inputs'; inputs.mkdir()
    for dataset in ['d4','d9']:
        shutil.copy2(args.baseline_root/f'inputs/{dataset}_events.jsonl',inputs/f'{dataset}_events.jsonl')
        shutil.copy2(args.baseline_root/f'{dataset}_one/warm_start.json',inputs/f'{dataset}_warm.json')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    manifest=dict(scope=__doc__,binary_sha256=sha256(binary),runs=[],
        input_hashes={p.name:sha256(p) for p in inputs.iterdir()},
        libraries=subprocess.check_output(['ldd',str(binary)],text=True))
    results={}
    for dataset,duration in [('d4',20),('d9',136)]:
        for budget in ['one','tight']:
            case=f'{dataset}_{budget}'; folder=output/case
            mode='joint_conditional_cached'+('_one' if budget=='one' else '')
            command=[str(binary),str(inputs/f'{dataset}_events.jsonl'),str(folder),mode,str(duration),
                '--warm-start',str(inputs/f'{dataset}_warm.json'),'--chart-audit','1:2','2:3','3:4','4:5']
            start=time.perf_counter()
            with (output/f'{case}.log').open('w') as log:
                try:
                    code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180).returncode
                except subprocess.TimeoutExpired:
                    code='timeout_180s'
            manifest['runs'].append(dict(case=case,command=command,exit_code=code,wall_seconds=time.perf_counter()-start))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if code!=0:
                raise RuntimeError(f'{case} failed; see log')
            summary=validate_ledger(folder)
            records=[json.loads(line) for line in (folder/'chart_audit.jsonl').read_text().splitlines()]
            verify_refresh_counts(folder,records)
            reference=args.baseline_root/case/'candidate1'
            comparison=compare_runs(reference,folder)
            comparison.pop('service_speedup'); comparison.pop('setup_inclusive_speedup')
            for filename in ['final_estimates.csv','causal_estimates.csv','control_messages.jsonl']:
                if sha256(reference/filename)!=sha256(folder/filename):
                    raise ValueError(f'{case} diagnostic changed {filename}')
            if any(comparison['frame_mismatches'].values()) or comparison['physical_cost_abs']:
                raise ValueError('diagnostic changed solver decisions')
            target=output/'reference'/case; target.mkdir(parents=True)
            for filename in ['summary.json','updates.csv','final_estimates.csv','causal_estimates.csv','control_messages.jsonl']:
                shutil.copy2(reference/filename,target/filename)
            results[case]=dict(summary=summary,comparison=comparison,audit=summarize_records(records))
            (output/'analysis.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(case=case,records=len(records),passive=True)),flush=True)


if __name__=='__main__':
    main()
