"""Fixed-chart mean/variance distinguishability audit, not an adaptive solver."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import svd
from scipy.stats import f as f_distribution

from experiments.network_noise_support import assemble

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dranppr_linear_evidence import LinearResidualEvidence


def range_directions(data):
    if (data.get('schema') != 'dranppr_physical_linearization_v1' or
            data.get('truth_used', False)):
        raise ValueError('invalid or truth-assisted physical export')
    records, rows = [], 0
    for factor in data['factors']:
        kind = factor['kind']
        if len(factor['b']) != {'observation': 2, 'odometry': 3}[kind]:
            raise ValueError('invalid factor dimension')
        if kind == 'observation':
            objects = [t for t in factor['terms'] if t['kind'] == 'l']
            if len(objects) != 1:
                raise ValueError('observation needs exactly one point')
            block = np.asarray(objects[0]['a'], dtype=float)
            if block.shape != (2, 2) or not np.isfinite(block).all():
                raise ValueError('invalid point Jacobian')
            # A range derivative with respect to a point has unit physical norm.
            precision_sqrt = float(np.linalg.norm(block[1]))
            if precision_sqrt <= 0:
                raise ValueError('zero range precision')
            records.append((rows+1, factor['robot'], objects[0]['index'], precision_sqrt))
        rows += len(factor['b'])
    if not records:
        raise ValueError('no range observations')
    robots = sorted({r for _, r, _, _ in records})
    objects = sorted({o for _, _, o, _ in records})
    names = [f'robot_{r}' for r in robots] + [f'object_{o}' for o in objects]
    columns = {name: i for i, name in enumerate(names)}
    directions = np.zeros((rows, len(names)))
    for row, robot, obj, precision in records:
        directions[row, columns[f'robot_{robot}']] = precision
        directions[row, columns[f'object_{obj}']] = precision
    return directions, names


def residual_geometry(a, directions, robot_columns):
    a, directions = np.asarray(a, float), np.asarray(directions, float)
    if (a.ndim != 2 or directions.ndim != 2 or len(a) != len(directions) or
            not np.isfinite(a).all() or not np.isfinite(directions).all() or
            not 0 < robot_columns < directions.shape[1]):
        raise ValueError('invalid design or offset directions')
    evidence = LinearResidualEvidence(a)
    c = evidence.contrast_basis
    t = c.T @ directions
    u, s, _ = svd(t[:, :robot_columns], full_matrices=False)
    # Use the original direction scale so numerical projection dust cannot
    # turn a wholly state-equivalent mean into a testable subspace.
    tolerance = max(a.shape+directions.shape)*np.finfo(float).eps*max(
        1., float(np.linalg.norm(directions[:, :robot_columns], 2)))
    rank = int(np.sum(s > tolerance))
    u = u[:, :rank]
    if not 0 < rank < c.shape[1]:
        raise ValueError('no testable robot subspace or residual complement')
    singular = svd(t, compute_uv=False)
    tol = max(a.shape+directions.shape)*np.finfo(float).eps*max(
        1., float(np.linalg.norm(directions, 2)))
    joint_rank = int(np.sum(singular > tol))
    norm = np.linalg.norm(a)
    diagnostic = dict(state_rank=evidence.rank, residual_dof=c.shape[1],
        robot_offset_rank=rank, joint_offset_rank=joint_rank,
        joint_offset_nullity=t.shape[1]-joint_rank,
        joint_offset_singular_values=singular.tolist(),
        contrast_annihilation_relative=float(np.linalg.norm(c.T @ a)/max(1., norm)),
        contrast_orthogonality_error=float(np.linalg.norm(c.T @ c-np.eye(c.shape[1]))),
        raw_offset_identity_error=float(np.max(np.abs(
            directions[:, :robot_columns].sum(axis=1)-directions[:, robot_columns:].sum(axis=1)))))
    return c, t, u, diagnostic


def sample_scores(z, u):
    z, u = np.asarray(z, float), np.asarray(u, float)
    if (z.ndim != 2 or u.ndim != 2 or len(z) != len(u) or
            not 0 < u.shape[1] < len(z) or not z.shape[1] or
            not np.isfinite(z).all() or not np.isfinite(u).all() or
            not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-12, rtol=1e-12)):
        raise ValueError('invalid residual samples or nonorthonormal subspace')
    mean_energy = np.sum((u.T @ z)**2, axis=0)
    total = np.sum(z*z, axis=0)
    residual = total-mean_energy
    if np.any(residual <= 0.):
        raise ValueError('nonpositive orthogonal energy; no flooring')
    q = residual/(len(z)-u.shape[1])
    return dict(scale_all=total/len(z), scale_orthogonal=q,
        mean_energy=mean_energy, f=(mean_energy/u.shape[1])/q)


def predictive_scores(train, check, u):
    if train.shape != check.shape:
        raise ValueError('training/check dimension mismatch')
    scores = sample_scores(train, u)
    fitted_mean = u @ (u.T @ train)
    def nll(innovation, q):
        return .5*(np.log(2*np.pi*q)+np.mean(innovation**2, axis=0)/q)
    return dict(scalar_nll=nll(check, scores['scale_all']),
        mean_nll=nll(check-fitted_mean, scores['scale_orthogonal']))


def simulate(t, names, u, samples, seed):
    if samples < 2 or t.shape != (len(u), len(names)):
        raise ValueError('invalid paired simulation dimensions')
    rng = np.random.default_rng(seed)
    train, check = rng.normal(size=(2, len(t), samples))
    zero = np.zeros(len(t))
    cases = [('gaussian_1', 1., zero), ('gaussian_4', 4., zero),
             ('state_equivalent', 1., zero)]
    for column, name in enumerate(names):
        for amplitude in (.10, .30):
            cases.append((f'{name}_{amplitude:.2f}m', 1., amplitude*t[:, column]))
    critical = float(f_distribution.ppf(.95, u.shape[1], len(u)-u.shape[1]))
    output = []
    for name, variance, mean in cases:
        one = np.sqrt(variance)*train+mean[:, None]
        two = np.sqrt(variance)*check+mean[:, None]
        score = sample_scores(one, u)
        prediction = predictive_scores(one, two, u)
        reject = float(np.mean(score['f'] > critical))
        delta = prediction['mean_nll']-prediction['scalar_nll']
        mean_projection = u @ (u.T @ mean)
        output.append(dict(case=name, true_variance=variance, samples=samples,
            expected_scale_all=float(variance+mean@mean/len(t)),
            expected_scale_orthogonal=float(variance+
                np.sum((mean-mean_projection)**2)/(len(t)-u.shape[1])),
            estimated_scale_all=float(np.mean(score['scale_all'])),
            estimated_scale_all_mc_se=float(np.std(score['scale_all'], ddof=1)/np.sqrt(samples)),
            estimated_scale_orthogonal=float(np.mean(score['scale_orthogonal'])),
            f_critical=critical, f_reject_fraction=reject,
            f_reject_mc_se=float(np.sqrt(reject*(1-reject)/samples)),
            scalar_nll=float(np.mean(prediction['scalar_nll'])),
            mean_nll=float(np.mean(prediction['mean_nll'])),
            mean_minus_scalar_nll=float(np.mean(delta)),
            paired_nll_mc_se=float(np.std(delta, ddof=1)/np.sqrt(samples))))
    return output


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def run(source, output):
    source, output = source.resolve(), output.resolve()
    hashes = {}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(source/name) != digest:
            raise ValueError('source checksum mismatch: '+name)
        hashes[name] = digest
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for name in ['experiments/audit_mean_evidence.py', 'experiments/network_noise_support.py',
                 'scripts/dranppr_linear_evidence.py', 'tests/test_mean_evidence.py',
                 'docs/superpowers/plans/2026-10-27-mean-evidence-audit.md']:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    manifest = dict(source=str(source), source_hashes=hashes, samples=4096, seed=20261027,
        command=sys.argv, truth_used=False, complete=False, runs=[],
        numpy_version=np.__version__,
        scope='Known fixed-chart Gaussian model only; all mean/scale comparisons are diagnostic, not an estimator or real-data fault classifier')
    write_json(output/'manifest.json', manifest)
    rows, details = [], []
    start = time.perf_counter()
    for dataset in (1, 4):
        linear = json.loads((source/f'd{dataset}/training/linearization.json').read_text())
        a, empirical, groups = assemble(linear)
        directions, names = range_directions(linear)
        nr = sum(n.startswith('robot_') for n in names)
        fit = json.loads((source/f'd{dataset}/fits/joint_qr/fit.json').read_text())
        for law, scales in [('nominal', np.ones(5)), ('fitted_as_known', np.asarray(fit['scales']))]:
            began = time.perf_counter()
            whitening = 1/np.sqrt(scales[groups])
            aw, bw = whitening[:, None]*a, whitening[:, None]*directions
            c, t, u, diagnostic = residual_geometry(aw, bw, nr)
            if diagnostic['contrast_annihilation_relative'] > 1e-10:
                raise ValueError('state directions not annihilated')
            if diagnostic['raw_offset_identity_error'] > 1e-12:
                raise ValueError('robot/object common-offset identity failed')
            # Explicitly verify the simulated invisible mean is an A*delta,
            # rather than assuming that every physical bias is invisible.
            delta = .01*np.sin(np.arange(aw.shape[1])+1.)
            state_mean = aw @ delta
            state_contrast = c.T @ state_mean
            state_ratio = np.linalg.norm(state_contrast)/max(1., np.linalg.norm(state_mean))
            if state_ratio > 1e-10:
                raise ValueError('state-equivalent mean survived projection')
            direction_rows = []
            for i, name in enumerate(names):
                energy = float(t[:, i] @ t[:, i])
                original = float(bw[:, i] @ bw[:, i])
                captured = float(np.sum((u.T @ t[:, i])**2))
                direction_rows.append(dict(name=name, range_observations=int(np.count_nonzero(bw[:, i])),
                    precision_after_state_elimination=energy,
                    retained_energy_fraction=energy/original,
                    fraction_in_robot_mean_subspace=captured/energy if energy > 1e-20 else None))
            z = c.T @ (whitening*empirical)
            empirical_score = sample_scores(z[:, None], u)
            result = dict(dataset=dataset, law=law, scales=scales.tolist(), shape=list(aw.shape),
                **diagnostic, state_equivalent_contrast_relative=float(state_ratio),
                state_equivalent_input_norm=float(np.linalg.norm(state_mean)), directions=direction_rows,
                empirical_descriptive_only={k: float(v[0]) for k, v in empirical_score.items()})
            sim = simulate(t, names, u, manifest['samples'], manifest['seed'])
            rows.extend(dict(dataset=dataset, law=law, **r) for r in sim)
            result['seconds'] = time.perf_counter()-began
            details.append(result)
            write_json(output/'geometry.json', details)
            write_json(output/'simulation.json', rows)
            manifest['runs'].append(dict(dataset=dataset, law=law, cases=len(sim), seconds=result['seconds']))
            write_json(output/'manifest.json', manifest)
            print(json.dumps(manifest['runs'][-1]), flush=True)
    with (output/'simulation.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(complete=True, total_seconds=time.perf_counter()-start)
    write_json(output/'manifest.json', manifest)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
