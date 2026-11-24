"""Posthoc transfer and injected-observation diagnostics; no estimator feedback."""
import argparse
import json
from pathlib import Path

from experiments.audit_conditional_replay import read_rows, sha256
from experiments.audit_lazy_refresh import dump
from experiments.analyze_full_lazy_refresh import recovery_diagnostics


def frame_status(rows, changes):
    identity = lambda r: (int(r['robot']), float(r['timestamp_s']))
    injected = {identity(c) for c in changes}
    if not injected <= {identity(r) for r in rows}:
        raise ValueError('injected observation frame missing from replay')
    groups = {name: dict(frames=0, accepted=0, requests=0, full_step_supported=0)
              for name in ['injected', 'not_injected']}
    for row in rows:
        result = groups['injected' if identity(row) in injected else 'not_injected']
        prediction = json.loads(row['full_step_prediction'])
        decrease = json.loads(row['full_step_decrease'])
        result['frames'] += 1
        result['accepted'] += int(row['accepted'])
        result['requests'] += int(row['refresh_requests'])
        result['full_step_supported'] += int(prediction is not None and prediction > 0 and
                                             decrease is not None and decrease > 0)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest['all_terminal'] or any(r['returncode'] != 0 for r in manifest['runs']):
        raise ValueError('unfinished or failed experiment')
    output = dict(cases={}, source_sha256=sha256(Path(__file__)),
        warning='Injected labels are posthoc diagnostics, not estimator inputs or an intended classifier. '
                'Physical costs on different input graphs are not comparable. One seed per stress condition.')
    clean = json.loads((root/'d2_clean/analysis.json').read_text())['results']['reference1']
    for case, kind in [('d2_clean', None), ('d3_clean', None),
                       ('d2_gaussian', 'gaussian_bearing'), ('d2_sparse', 'sparse_bearing')]:
        folder = root/case
        manifest = json.loads((folder/'manifest.json').read_text())
        if not manifest['all_estimators_terminal'] or any(r['exit_code'] != 0 for r in manifest['runs']):
            raise ValueError('unfinished suffix comparison')
        data = json.loads((folder/'analysis.json').read_text())
        changes = []
        stress = None
        if kind:
            stress = json.loads((root/'inputs'/kind/'stress_manifest.json').read_text())
            if (stress['events_sha256'] != manifest['events_sha256'] or
                    stress['warm_start_sha256'] != manifest['warm_start_sha256'] or
                    not stress['prefix_byte_identical']):
                raise ValueError('stress input provenance mismatch')
            changes = stress['changes']
            for key in ['frames', 'packets', 'physical_factors', 'prefix_physical_factors', 'initial_cost']:
                if clean[key] != data['results']['reference1'][key]:
                    raise ValueError(f'clean/stress mismatch: {key}')
        runs = {}
        for tag, result in data['results'].items():
            rows = read_rows(folder/tag/'updates.csv')
            runs[tag] = dict(recovery=recovery_diagnostics(rows),
                frame_status=frame_status(rows, changes),
                accumulated_accepted_loss_decrease=sum(float(r['cost_before'])-float(r['cost_after']) for r in rows),
                frames_with_positive_accepted_loss_decrease=sum(float(r['cost_after']) < float(r['cost_before']) for r in rows))
        reference, candidate = data['results']['reference1'], data['results']['candidate1']
        output['cases'][case] = dict(runs=runs, input_stress=stress,
            source_analysis_sha256=sha256(folder/'analysis.json'),
            paired_speedups=[data['results'][f'reference{i}']['service_seconds']/
                             data['results'][f'candidate{i}']['service_seconds'] for i in [1, 2]],
            communication_increase_percent=100*(candidate['application_bytes']/reference['application_bytes']-1),
            physical_cost_increase_percent=100*(candidate['physical_cost']/reference['physical_cost']-1),
            trajectory_rmse_change_m=candidate['quality']['translation_rmse_m']-reference['quality']['translation_rmse_m'],
            object_rmse_change_m=candidate['quality']['object_copy_rmse_m']-reference['quality']['object_copy_rmse_m'],
            causal_rmse_change_m=candidate['causal_quality']['translation_rmse_m']-reference['causal_quality']['translation_rmse_m'])
    dump(root/'stress_diagnostics.json', output)
    for case, result in output['cases'].items():
        print(json.dumps(dict(case=case, **{k:v for k,v in result.items() if k not in ['runs', 'input_stress']})))
        print(json.dumps(result['runs']['candidate1']))


if __name__ == '__main__':
    main()
