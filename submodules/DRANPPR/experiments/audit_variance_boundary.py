"""Conditional variance-boundary audit, not a Bayesian SLAM estimator.

Hold nuisance components and the local linearization fixed. Compare truncated
log-uniform and one-dimensional Jeffreys slices only to expose prior/cutoff
dependence. Neither slice is a joint parameter posterior or confidence interval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.linalg import cho_factor, cho_solve


def slice_terms(z, base, component, logs):
    z, base, component, logs = [np.asarray(x, float) for x in [z, base, component, logs]]
    if (z.ndim != 1 or not len(z) or base.shape != (len(z),len(z)) or
            component.shape != base.shape or logs.ndim != 1 or not len(logs) or
            not all(np.isfinite(x).all() for x in [z,base,component,logs]) or
            not np.allclose(base, base.T) or not np.allclose(component, component.T)):
        raise ValueError('invalid variance slice')
    if not np.any(component):
        raise ValueError('uninformative zero variance component')
    nll, log_jeffreys = [], []
    for log_q in logs:
        factor = cho_factor(base+np.exp(log_q)*component, lower=True, check_finite=False)
        solved_z = cho_solve(factor, z, check_finite=False)
        nll.append(.5*(z@solved_z+2*np.log(np.diag(factor[0])).sum()+len(z)*np.log(2*np.pi)))
        derivative = cho_solve(factor, component, check_finite=False)
        fisher_q = .5*np.einsum('ij,ji->', derivative, derivative)
        if not np.isfinite(fisher_q) or fisher_q <= 0:
            raise ValueError('uninformative or numerically invalid Fisher information')
        log_jeffreys.append(.5*np.log(fisher_q))
    return np.array(nll), np.array(log_jeffreys)


def truncated_summary(logs, log_density_in_log_coordinate):
    logs, weights = np.asarray(logs, float), np.asarray(log_density_in_log_coordinate, float)
    if (logs.ndim != 1 or len(logs)<3 or weights.shape != logs.shape or
            not np.all(np.diff(logs)>0) or not np.isfinite(weights).all()):
        raise ValueError('invalid quadrature grid')
    shift = float(weights.max())
    density = np.exp(weights-shift)
    integral = float(simpson(density, x=logs))
    cdf = cumulative_trapezoid(density, x=logs, initial=0.)
    cdf /= cdf[-1]
    return dict(log_integral=shift+np.log(integral),
                mean_component=float(simpson(density*np.exp(logs), x=logs)/integral),
                median_component=float(np.exp(np.interp(.5, cdf, logs))),
                lower_log_bound=float(logs[0]), upper_log_bound=float(logs[-1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--components', nargs='+', type=int, default=[3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts'))
    from audit_contrast_noise import calibrate_contrast
    from audit_online_lower import replay_fixed_calibration
    from audit_residual_evidence import write_csv
    from dranppr_variance_components import ContrastVarianceModel
    producer = json.loads((args.run/'manifest.json').read_text())
    events = args.stream/'events.jsonl'
    if digest(events) != producer['event_sha256']:
        raise ValueError('event digest mismatch')
    with events.open() as stream:
        robots, _ = replay_fixed_calibration((json.loads(line) for line in stream), [],
                                             producer['warmup_s'], producer['warmup_s'], ('fixed',))
    fit, (a,residual,d,base,scale) = calibrate_contrast(robots['fixed'][args.robot], 'reml_diagonal_q')
    expected = next(row for row in json.loads((args.run/'calibration.json').read_text())
                    if row['method']=='reml_diagonal_q' and row['robot']==args.robot)
    if not np.allclose(fit['components'], expected['components'], rtol=1e-8, atol=1e-12):
        raise ValueError('warmup reconstruction differs from saved run')
    model = ContrastVarianceModel(a, -residual, d, base)
    q = np.asarray(fit['components'])
    names = ['q_longitudinal','q_lateral','q_yaw','bearing','range_intercept','range_slope']
    logs = np.linspace(-64., 0., 4097)
    summary, grids = [], []
    for index in args.components:
        if not 0 <= index < len(q):
            raise ValueError('invalid component index')
        other = q.copy()
        other[index] = 0.
        zero_covariance = model.base + np.einsum('k,kij->ij', other, model.components)
        eigenvalues = np.linalg.eigvalsh(zero_covariance)
        nll, jeffreys = slice_terms(model.z, zero_covariance, model.components[index], logs)
        numerical_spd = eigenvalues[0] > 64*np.finfo(float).eps*max(1.,eigenvalues[-1])
        boundary = dict(component=names[index], index=index, nuisance_fixed=True,
                        zero_covariance_min_eigenvalue=float(eigenvalues[0]),
                        zero_covariance_max_eigenvalue=float(eigenvalues[-1]),
                        zero_covariance_numerically_spd=bool(numerical_spd),
                        nll_at_log_minus64=float(nll[0]), nll_at_log_minus16=float(nll[3072]),
                        log_uniform_nonintegrability_condition_satisfied=bool(numerical_spd))
        for prior, log_density in [('log_uniform',-nll),
                                   ('conditional_jeffreys',-nll+jeffreys+logs)]:
            # Jeffreys above is a density in q; +logs is the dq/dlog(q) Jacobian.
            for lower in [-16.,-32.,-64.]:
                mask = logs>=lower
                summary.append(dict(**boundary, prior=prior,
                                    **truncated_summary(logs[mask], log_density[mask])))
        grids.extend(dict(component=names[index], log_component=float(t), nll=float(n),
                          log_jeffreys_density_in_q=float(j)) for t,n,j in zip(logs,nll,jeffreys))
    write_csv(args.output/'summary.csv', summary)
    write_csv(args.output/'slices.csv', grids)
    np.savez(args.output/'contrast_model.npz', z=model.z, base=model.base,
             components=model.components, fitted_components=q)
    (args.output/'calibration.json').write_text(json.dumps(fit, indent=2)+'\n')
    sources = [Path(__file__), root/'tests/test_variance_boundary_audit.py',
               root/'experiments/audit_contrast_noise.py',root/'scripts/dranppr_variance_components.py',
               root/'experiments/audit_online_lower.py',root/'scripts/dranppr_incremental.py']
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/path.name)
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    source_hashes={p.name:digest(p) for p in sources},
                    producer_manifest_sha256=digest(args.run/'manifest.json'),
                    event_sha256=digest(events), truth_used=False, warmup_only=True,
                    postselected=True, state_feedback=False, conditional_not_joint=True,
                    confidence_intervals=False, proper_joint_posterior_claim=False,
                    scope='Lower truncation sensitivity at fixed nuisance ML values and fixed linearization',
                    numerical_grid='4097 equally spaced log points [-64,0]; lower cutoffs -16,-32,-64; fixed upper cutoff 0',
                    distance_scale=scale)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
