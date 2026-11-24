"""Read-only model-response review; writes a separate sealed evidence report."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.async_interface import Scenario
from experiments.audit_async_interface import ROOT, digest, write_json
from experiments.audit_error_budget import audit_updates
from experiments.audit_model_response import at_common_bytes, audit_bytes, summarize
from experiments.audit_response_controls import load_model, seal
from experiments.review_async_interface import verify_ledger, gap_comparison
from experiments.review_error_budget import error_coverage


def review(root, parity_root=None):
    manifest = verify_ledger(root)
    records = json.loads((root/'runs.json').read_text())
    expected = len(manifest['cases'])*len(manifest['scenarios'])*len(manifest['seeds'])*len(manifest['arms'])
    assert len(records) == len({(r['case'], r['scenario'], r['seed'], r['arm']) for r in records}) == expected
    if parity_root is not None:
        verify_ledger(parity_root)
    checks, comparisons, parity = [], [], 0
    for case in manifest['cases']:
        model, links = load_model(root, case)
        initial = np.load(root/case/'initial.npy')
        optimum = cho_solve(cho_factor(model.b, lower=True), -model.c)
        start = initial-optimum
        for net in manifest['scenarios']:
            for seed in manifest['seeds']:
                scenario = (Scenario(net, 12, 20, seed=seed) if net == 'delay' else
                    Scenario(net, seed=seed, outage=(200, 600))) if manifest['stress'] else Scenario(net, seed=seed)
                runs = {}
                for arm in manifest['arms']:
                    name = f'{net}_{seed}_{arm}'
                    row = json.loads((root/case/(name+'.json')).read_text())
                    with np.load(root/case/(name+'.npz')) as data:
                        run = {key: data[key] for key in data.files}
                    run['summary'] = row
                    error = run['states']-optimum
                    gaps = np.einsum('ij,ij->i', error@model.b, error)/(start@model.b@start)
                    np.testing.assert_allclose(gaps, run['relative_gaps'], rtol=1e-5, atol=1e-20)
                    assert row['relative_gap'] == run['relative_gaps'][-1]
                    update, traffic = audit_updates(model, run, initial), audit_bytes(model, links, scenario, run)
                    checked = dict(case=case, scenario=net, seed=seed, arm=arm, update=update, traffic=traffic)
                    if arm != 'baseline':
                        checked['coverage'] = error_coverage(model, initial, run)
                    if parity_root is not None and arm in ('baseline', 'budget_hold', 'budget_predict'):
                        with np.load(parity_root/case/(name+'.npz')) as old:
                            np.testing.assert_array_equal(run['states'], old['states'])
                            np.testing.assert_array_equal(run['counters'], old['counters'])
                        parity += 1
                    if net == 'ideal' and runs:
                        np.testing.assert_array_equal(run['states'], runs['baseline']['states'])
                    runs[arm] = run
                    checks.append(checked)
                np.testing.assert_array_equal(runs['model_response']['counters'], runs['model_frozen']['counters'])
                for other in ('baseline', 'budget_hold', 'budget_predict', 'model_frozen'):
                    candidate, reference = runs['model_response'], runs[other]
                    aligned = at_common_bytes(candidate, reference)
                    without_setup = dict(candidate, counters=candidate['counters'].copy())
                    without_setup['counters'][:, 0] -= candidate['summary']['model_setup_bytes']
                    comparison = dict(case=case, scenario=net, seed=seed, name='model_response_vs_'+other,
                        alignment=aligned, equal_bytes=aligned['result'],
                        equal_ticks=gap_comparison(candidate['relative_gaps'][-1], reference['relative_gaps'][-1]),
                        equal_bytes_no_candidate_setup=at_common_bytes(without_setup, reference),
                        payload_ratio=float(candidate['counters'][-1, 0]/reference['counters'][-1, 0]))
                    comparisons.append(comparison)
    return dict(summary=summarize(records, comparisons), checks=checks, comparisons=comparisons,
                exact_legacy_pairs=parity, source_ledger=digest(root/'SHA256SUMS'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parity-root', type=Path)
    args = parser.parse_args()
    result = review(args.root, args.parity_root)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('review_model_response', 'audit_model_response', 'review_error_budget', 'audit_error_budget'):
        shutil.copy2(ROOT/'experiments'/(name+'.py'), args.output/(name+'.py'))
    write_json(args.output/'review.json', result)
    seal(args.output, dict(all_terminal=False, source_root=str(args.root.resolve()),
        source_ledger=result['source_ledger'], parity_root=str(args.parity_root),
        scope='Read-only integer-traffic, update, metric and legacy replay audit'))
    print(json.dumps(dict(result['summary'], exact_legacy_pairs=result['exact_legacy_pairs'])), flush=True)


if __name__ == '__main__':
    main()
