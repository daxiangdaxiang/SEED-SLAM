"""Posthoc temporal, recovery and historical-context audit for full suffixes."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from experiments.audit_conditional_replay import read_rows, sha256
from experiments.audit_lazy_refresh import dump


def _longest_run(flags):
    longest = current = 0
    for flag in flags:
        current = current+1 if flag else 0
        longest = max(longest, current)
    return longest


def recovery_diagnostics(rows):
    requests, stalled = [], []
    for index, row in enumerate(rows):
        event = dict(frame=index, elapsed_s=float(row['elapsed_s']), accepted=int(row['accepted']))
        next_row = rows[index+1] if index+1 < len(rows) else None
        event.update(next_accepted=int(next_row['accepted']) if next_row else None,
                     next_refresh=int(next_row['refresh_rounds']) if next_row else None,
                     next_request=int(next_row['refresh_requests']) if next_row else None)
        if int(row['refresh_requests']):
            requests.append(event)
        if not int(row['accepted']):
            stalled.append(event)
    return dict(stalled_frames=len(stalled), stalled_events=stalled, request_events=requests,
        longest_stalled_run_frames=_longest_run(not int(r['accepted']) for r in rows),
        longest_repeated_refresh_requests=_longest_run(int(r['refresh_requests']) for r in rows),
        next_frame_full_step_recoveries=sum(e['next_accepted'] == 1 and e['next_refresh'] == 1 and
                                          e['next_request'] == 0 for e in requests),
        requests_without_next_frame=sum(e['next_accepted'] is None for e in requests))


def temporal_profile(rows, squared_errors):
    if len(rows) != len(squared_errors) or not rows:
        raise ValueError('trace/error length mismatch or empty trace')
    if any(e is not None and (not math.isfinite(e) or e < 0) for e in squared_errors):
        raise ValueError('invalid squared error')
    output = []
    for quarter in range(4):
        first, stop = quarter*len(rows)//4, (quarter+1)*len(rows)//4
        block = rows[first:stop]
        if not block:
            continue
        errors = [e for e in squared_errors[first:stop] if e is not None]
        service = np.array([float(r['total_ms'])/1000 for r in block])
        output.append(dict(quarter=quarter+1, first_frame=first, stop_frame_exclusive=stop,
            frames=len(block), supported_poses=len(errors), elapsed_start_s=float(block[0]['elapsed_s']),
            elapsed_stop_s=float(block[-1]['elapsed_s']),
            measurement_rmse_m=math.sqrt(sum(errors)/len(errors)) if errors else None,
            service_seconds=float(service.sum()), frame_p50_s=float(np.quantile(service,.5)),
            frame_p95_s=float(np.quantile(service,.95)), frame_max_s=float(service.max()),
            backlog_max_s=max(float(r['serial_backlog_s']) for r in block),
            accepted_frames=sum(int(r['accepted']) for r in block),
            refresh_requests=sum(int(r['refresh_requests']) for r in block),
            application_bytes=sum(int(r['optimizer_bytes'])+int(r['control_bytes']) for r in block)))
    return output


def aligned_squared_errors(causal, truth, alignment):
    c, s = math.cos(alignment['alignment_yaw']), math.sin(alignment['alignment_yaw'])
    rotation = np.array([[c,-s],[s,c]])
    translation = np.array([alignment['alignment_x'],alignment['alignment_y']])
    errors = []
    for row in causal:
        t, series = float(row['timestamp_s']), truth.robots[int(row['robot'])]
        if not series.supports(t):
            errors.append(None)
            continue
        point = rotation @ [float(row['x']),float(row['y'])]+translation
        errors.append(float(np.sum((point-series.interpolate(t)[0])**2)))
    supported = [e for e in errors if e is not None]
    if not math.isclose(math.sqrt(sum(supported)/len(supported)), alignment['translation_rmse_m'],
                        rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('whole-suffix alignment/error mismatch')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('analysis output already exists')
    root = Path(__file__).resolve().parents[1]
    cases = {}
    for dataset in [4,9]:
        folder = args.results/f'd{dataset}'
        manifest = json.loads((folder/'manifest.json').read_text())
        if not manifest['all_estimators_terminal'] or any(r['exit_code'] != 0 for r in manifest['runs']):
            raise ValueError('unfinished or failed replay; do not rank partial graphs')
        cases[dataset] = dict(folder=folder, manifest=manifest,
                             analysis=json.loads((folder/'analysis.json').read_text()))
    from benchmarks.mrclam.evaluate import load_ground_truth
    from benchmarks.mrclam.source import audit_dataset, load_manifest
    output = dict(scope='Posthoc full-suffix audit, unchanged estimator. Quarters split by frame count; one whole-suffix alignment, no per-quarter realignment.',
                  truth_used_by_estimator=False, cases={})
    for dataset, case in cases.items():
        folder, manifest, analysis = case['folder'], case['manifest'], case['analysis']
        data_manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audit = audit_dataset(data_manifest,args.raw)
        truth = load_ground_truth(data_manifest,audit.dataset_directory)
        case_result = dict(runs={}, historical_context={})
        for tag, result in analysis['results'].items():
            updates = read_rows(folder/tag/'updates.csv')
            causal = read_rows(folder/tag/'causal_estimates.csv')
            identities = lambda rows: [(int(r['robot']),float(r['timestamp_s'])) for r in rows]
            if identities(updates) != identities(causal):
                raise ValueError('causal states and service trace do not align')
            errors = aligned_squared_errors(causal,truth,result['causal_quality'])
            case_result['runs'][tag] = dict(recovery=recovery_diagnostics(updates),
                                           temporal=temporal_profile(updates,errors))
        historical = root/f'results/dranppr_native_online_20260923/d{dataset}_one'
        old_manifest = json.loads((historical/'manifest.json').read_text())
        if (old_manifest['package_sha256'] != manifest['warm_start_sha256'] or
                old_manifest['event_sha256'] != manifest['events_sha256']):
            raise ValueError('historical baseline inputs differ')
        old = json.loads((historical/'comparison.json').read_text())
        reference = analysis['results']['candidate1']
        updates = read_rows(folder/'candidate1/updates.csv')
        credits, total = [], 0
        for row in updates:
            total += int(row['optimizer_bytes'])+int(row['control_bytes'])
            credits.append(total)
        for result in old['results']:
            tag = result['tag']
            for key in ['frames','physical_factors','prefix_physical_factors']:
                if result[key] != reference[key]:
                    raise ValueError(f'historical {tag}: {key} mismatch')
            if not math.isclose(result['initial_cost'],reference['initial_cost'],rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError(f'historical {tag}: initial physical cost mismatch')
            old_updates = read_rows(historical/tag/'updates.csv')
            if identities(updates) != identities(old_updates):
                raise ValueError('historical arrived frame identities differ')
            budget_check = None
            if tag != 'dran':
                trace = [json.loads(line) for line in (historical/tag/'trace.jsonl').open()]
                if identities(trace) != identities(updates):
                    raise ValueError('historical byte trace identities differ')
                differences = [r['application_bytes']-credit for r,credit in zip(trace,credits)]
                budget_check = dict(frames_exceeding_candidate_cumulative_bytes=sum(d > 0 for d in differences),
                                    maximum_excess_bytes=max(0,max(differences)),
                                    terminal_unused_bytes=credits[-1]-trace[-1]['application_bytes'])
            case_result['historical_context'][tag] = dict(result, candidate_budget_check=budget_check)
        case_result['historical_context_warning'] = ('Archived results, NOT rerun timing or a new byte-filling schedule. '
            'Exact input hashes and arrival frames checked; old byte-ceiling schedules retain their original policy. '
            'Baselines with disagreeing copies have infeasible physical costs for the exact-consensus problem.')
        case_result['provenance'] = dict(historical_manifest_sha256=sha256(historical/'manifest.json'),
            historical_comparison_sha256=sha256(historical/'comparison.json'),
            current_analysis_sha256=sha256(folder/'analysis.json'))
        output['cases'][str(dataset)] = case_result
    dump(args.output,output)
    snapshot = args.results/'posthoc_source'
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    shutil.copy2(root/'tests/test_analyze_full_lazy_refresh.py',snapshot/'test_analyze_full_lazy_refresh.py')
    print(json.dumps({dataset:{tag:r['recovery']['stalled_frames'] for tag,r in case['runs'].items()}
                      for dataset,case in output['cases'].items()}))


if __name__ == '__main__':
    main()
