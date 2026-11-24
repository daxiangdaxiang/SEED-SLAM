"""Independent trace/parity review for the 2026-11-23 controlled solver study."""
import argparse
import json
from pathlib import Path

from experiments.audit_mean_evidence import sha256,write_json
from experiments.audit_rigid_subspace import budget_audit
from experiments.evaluate_incremental_reference import read_rows


def compare_traces(left, right):
    result = budget_audit(left,right)
    fields = ('robot','timestamp_s','cost_before','cost_after','accepted',
              'refresh_requests','refresh_rounds','refreshed_keys','factors_linearized',
              'full_step_prediction','full_step_decrease','pcg_iterations')
    for a,b in zip(left,right):
        if any(a[k]!=b[k] for k in fields):
            raise ValueError('physical trace changed')
    result['max_residual_discrepancy'] = max(abs(float(a['pcg_relative_residual'])-
        float(b['pcg_relative_residual'])) for a,b in zip(left,right))
    return result


def summarize(root, output):
    records, coarse = [], []
    for suffix in ('d14','d23','d9'):
        folder=root/f'results/dranppr_fused_{suffix}_20261123'
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest['all_estimators_terminal'] or any(r['exit_code']!=0 for r in manifest['runs']):
            raise ValueError('incomplete experiment')
        source=Path(manifest['source'])
        quality=json.loads((source/'comparison.json').read_text())['records']
        for case in manifest['benchmark_cases']:
            dataset=case['dataset']
            for weights in ('nominal','calibrated'):
                a,b=[folder/f'd{dataset}/{weights}_{solver}' for solver in ('pcg','fused')]
                for name in ('final_estimates.csv','causal_estimates.csv'):
                    if len({sha256(p/name) for p in (a,b,source/f'd{dataset}/{weights}')})!=1:
                        raise ValueError('state parity failure')
                left,right=[json.loads((p/'summary.json').read_text()) for p in (a,b)]
                trace=compare_traces(read_rows(a/'updates.csv'),read_rows(b/'updates.csv'))
                if trace['prefix_exceedances'] or right['physical_cost']!=left['physical_cost']:
                    raise ValueError('claimed dominance fails')
                q=next(r for r in quality if r['dataset']==dataset and r['arm']==weights)
                records.append(dict(dataset=dataset,weights=weights,trace=trace,
                    bytes_before=left['application_bytes'],bytes_after=right['application_bytes'],
                    reduction_percent=100*(1-right['application_bytes']/left['application_bytes']),
                    seconds_before=left['service_seconds'],seconds_after=right['service_seconds'],
                    service_change_percent=100*(right['service_seconds']/left['service_seconds']-1),
                    causal_rmse_m=q['causal_quality']['translation_rmse_m'],
                    terminal_rmse_m=q['quality']['translation_rmse_m'],
                    object_rmse_m=q['quality']['object_copy_rmse_m'],
                    quality_provenance='Archived evaluation of byte-identical outputs, not a fresh baseline run.',
                    baseline_source=str(source),physical_cost=right['physical_cost']))
    for suffix in ('d1','d23','d4','d9'):
        folder=root/f'results/dranppr_rigid_{suffix}_20261123'
        comparison=json.loads((folder/'comparison.json').read_text())['records']
        fixed=json.loads((folder/'fixed_frame.json').read_text())['records']
        for r in comparison:
            if not r['arm'].endswith('_rigid'):
                continue
            d,weight=r['dataset'],r['arm'].removesuffix('_rigid')
            a=next(p for p in comparison if p['dataset']==d and p['arm']==weight+'_pcg')
            old=next(p for p in fixed if p['dataset']==d and p['arm']==a['arm'])
            new=next(p for p in fixed if p['dataset']==d and p['arm']==r['arm'])
            coarse.append(dict(dataset=d,weights=weight,
                causal_before=a['causal_quality']['translation_rmse_m'],
                causal_after=r['causal_quality']['translation_rmse_m'],
                fixed_frame_before=old['rmse_m'],fixed_frame_after=new['rmse_m'],
                terminal_before=a['quality']['translation_rmse_m'],terminal_after=r['quality']['translation_rmse_m'],
                cost_before=a['summary']['physical_cost'],cost_after=r['summary']['physical_cost'],
                bytes_before=a['summary']['application_bytes'],bytes_after=r['summary']['application_bytes']))
    if len(records)!=10 or len(coarse)!=10:
        raise ValueError('missing declared paired conditions')
    result=dict(fused=sorted(records,key=lambda r:(r['dataset'],r['weights'])),
        coarse=sorted(coarse,key=lambda r:(r['dataset'],r['weights'])),
        script_sha256=sha256(__file__), no_truth_used_to_choose_fused_updates=True,
        scope='Communication-equivalent state update milestone, NOT general acceleration or noise robustness.')
    write_json(output,result)
    for r in result['fused']:
        print(f"D{r['dataset']} {r['weights']}: {r['bytes_before']/1e6:.6f} -> {r['bytes_after']/1e6:.6f} MB, "
              f"-{r['reduction_percent']:.3f}%, service {r['service_change_percent']:+.3f}%",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(Path(__file__).resolve().parents[1],args.output)
