"""Post-selected warmup likelihood profiles, not an adaptation acceptance rule.

Profiles diagnose support within the assumed fixed-linear Gaussian family.
No chi-square confidence coverage is claimed at variance boundaries.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

from audit_contrast_noise import calibrate_contrast
from audit_online_lower import replay_fixed_calibration
from audit_residual_evidence import sha256, write_csv
from dranppr_variance_components import ContrastVarianceModel


def profile_component(model, index, value, initial):
    if not np.isfinite(value) or value <= 0.:
        raise ValueError('profiled component must be positive')
    initial = np.asarray(initial, dtype=float).copy()
    if initial.shape != (len(model.components),) or not 0 <= index < len(initial):
        raise ValueError('invalid profile component or initial vector')
    initial[index] = np.log(value)
    free = np.arange(len(initial)) != index
    def expand(nuisance):
        theta = initial.copy()
        theta[free] = nuisance
        return theta
    if not np.any(free):
        return dict(components=np.exp(initial).tolist(), nll=model.nll(initial), success=True)
    result = minimize(lambda x: model.nll(expand(x)), initial[free],
                      jac=lambda x: model.gradient(expand(x))[free], method='L-BFGS-B',
                      bounds=[(-16., 10.)]*int(np.sum(free)),
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7})
    return dict(components=np.exp(expand(result.x)).tolist(), nll=float(result.fun),
                success=bool(result.success), message=str(result.message))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('output must be empty')
    producer = json.loads((args.run/'manifest.json').read_text())
    events = args.stream/'events.jsonl'
    if sha256(events) != producer['event_sha256']:
        raise ValueError('event digest mismatch')
    with events.open() as stream:
        robots, _ = replay_fixed_calibration((json.loads(line) for line in stream), [],
                    producer['warmup_s'], producer['warmup_s'], ('fixed',))
    fit, (a, residual, d, base, scale) = calibrate_contrast(robots['fixed'][args.robot], 'reml_diagonal_q')
    expected = next(row for row in json.loads((args.run/'calibration.json').read_text())
                    if row['method'] == 'reml_diagonal_q' and row['robot'] == args.robot)
    if not np.allclose(fit['components'], expected['components'], rtol=1e-8, atol=1e-12):
        raise ValueError('warmup calibration reconstruction mismatch')
    model = ContrastVarianceModel(a, -residual, d, base)
    rows, details = [], []
    names = ('q_longitudinal', 'q_lateral', 'q_yaw', 'bearing', 'range_intercept', 'range_slope')
    for index, name in enumerate(names):
        for value in np.unique(np.r_[np.exp(np.linspace(-16., 0., 17)), fit['components'][index]]):
            candidates = [profile_component(model, index, value, initial) for initial in
                          (np.log(fit['components']), np.log([1., 1., 1., 1., .5, .5]))]
            successful = [candidate for candidate in candidates if candidate['success']]
            if not successful:
                raise RuntimeError(f'profile optimizers failed: {name}: {value}: {candidates}')
            best = min(successful, key=lambda candidate: candidate['nll'])
            row = dict(robot=args.robot, component=name, multiplier=float(value),
                       profile_nll=best['nll'], delta_nll=best['nll']-fit['nll'],
                       successful_starts=len(successful))
            rows.append(row)
            details.append(dict(**row, nuisance_fits=candidates))
    write_csv(args.output/'profiles.csv', rows)
    (args.output/'profiles.json').write_text(json.dumps(details, indent=2)+'\n')
    (args.output/'calibration.json').write_text(json.dumps(fit, indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'experiments/audit_contrast_noise.py',
               root/'experiments/audit_online_lower.py', root/'scripts/dranppr_variance_components.py']
    (args.output/'manifest.json').write_text(json.dumps(dict(
        argv=sys.argv, source_hashes={str(p.relative_to(root)): sha256(p) for p in sources},
        producer_manifest_sha256=sha256(args.run/'manifest.json'), event_sha256=sha256(events),
        truth_used=False, postselected=True, warmup_only=True, distance_scale=scale,
        confidence_intervals=False, fixed_linearization=True), indent=2)+'\n')
    print(json.dumps([row for row in rows if row['multiplier'] == 1.], indent=2))


if __name__ == '__main__':
    main()
