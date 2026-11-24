"""Fixed physical quadratics for an asynchronous-interface diagnostic.

Dense reduced rows are pre-distributed for this experiment. This module is
setup/evaluation code, not a deployed distributed model-assembly protocol.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import splu


@dataclass
class Elimination:
    h: np.ndarray
    g: np.ndarray
    private_h: object
    private_cross: np.ndarray
    private_rhs: np.ndarray
    factor: object

    def recover(self, boundary):
        return self.factor.solve(self.private_rhs-self.private_cross@boundary)

    def private_residual(self, boundary):
        rhs = self.private_rhs-self.private_cross@boundary
        return float(np.linalg.norm(self.private_h@self.recover(boundary)-rhs)/
                     max(np.linalg.norm(rhs), np.finfo(float).tiny))


def eliminate(private, boundary, rhs):
    private, boundary = sparse.csc_matrix(private), np.asarray(boundary, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if (private.shape[0] != len(rhs) or boundary.shape[0] != len(rhs) or
            not np.isfinite(private.data).all() or not np.isfinite(boundary).all() or
            not np.isfinite(rhs).all()):
        raise ValueError('invalid local Jacobian')
    hp = (private.T@private).tocsc()
    cross, bp = np.asarray(private.T@boundary), np.asarray(private.T@rhs)
    try:
        factor = splu(hp)
    except RuntimeError as error:
        raise ValueError('singular private model; no prior or ridge added') from error
    h = boundary.T@boundary-cross.T@factor.solve(cross)
    g = -boundary.T@rhs+cross.T@factor.solve(bp)
    return Elimination(.5*(h+h.T), g, hp, cross, bp, factor)


@dataclass
class NormalizedModel:
    h: np.ndarray
    g: np.ndarray
    b: np.ndarray
    c: np.ndarray
    transform: np.ndarray
    indices: dict
    condition: float

    def to_physical(self, y):
        return self.transform@y

    def rows(self):
        return {i: {j: self.b[np.ix_(ii, jj)].copy()
                    for j, jj in self.indices.items()
                    if np.any(self.b[np.ix_(ii, jj)] != 0.)}
                for i, ii in self.indices.items()}


def normalize(h, g, owners):
    h, g, owners = np.asarray(h, float), np.asarray(g, float), np.asarray(owners)
    if (h.shape != (len(g), len(g)) or owners.shape != g.shape or
            not np.isfinite(h).all() or not np.isfinite(g).all() or
            not np.allclose(h, h.T, rtol=1e-12, atol=1e-12)):
        raise ValueError('invalid symmetric quadratic')
    try:
        np.linalg.cholesky(h)
    except np.linalg.LinAlgError as error:
        raise ValueError('physical interface is not positive definite') from error
    indices = {int(i): np.flatnonzero(owners == i) for i in sorted(set(owners))}
    transform = np.zeros_like(h)
    for indices_i in indices.values():
        local = np.linalg.cholesky(h[np.ix_(indices_i, indices_i)])
        transform[np.ix_(indices_i, indices_i)] = solve_triangular(
            local.T, np.eye(len(indices_i)), lower=False)
    b, c = transform.T@h@transform, transform.T@g
    return NormalizedModel(h.copy(), g.copy(), .5*(b+b.T), c, transform,
                           indices, float(np.linalg.cond(b)))


def physical_factors(data):
    """Read archived Pose2/Point2 factors and eliminate private poses locally."""
    if data.get('schema') != 'dranppr_physical_linearization_v1' or data.get('truth_used'):
        raise ValueError('unsupported physical factor schema')
    factors = data['factors']
    robots = sorted({f['robot'] for f in factors})
    labels = sorted({t['index'] for f in factors for t in f['terms'] if t['kind'] == 'l'})
    columns = {key: 2*k for k, key in enumerate(labels)}
    observers = {key: sorted({f['robot'] for f in factors for t in f['terms']
                             if t['kind'] == 'l' and t['index'] == key}) for key in labels}
    # Static, reproducible ownership balances labels without changing identity.
    owners = [observers[key][k % len(observers[key])] for k, key in enumerate(labels) for _ in range(2)]
    local_models, full_residual = [], 0.
    h, g = np.zeros((2*len(labels), 2*len(labels))), np.zeros(2*len(labels))
    for robot in robots:
        local = [f for f in factors if f['robot'] == robot]
        poses = sorted({t['index'] for f in local for t in f['terms'] if t['kind'] == 'x'})
        pc = {key: 3*k for k, key in enumerate(poses)}
        rows = sum(len(f['b']) for f in local)
        ab, rhs = np.zeros((rows, len(g))), np.zeros(rows)
        ri, ci, entries, offset = [], [], [], 0
        for f in local:
            size = len(f['b'])
            rhs[offset:offset+size] = f['b']
            for term in f['terms']:
                block = np.asarray(term['a'], float)
                expected = 3 if term['kind'] == 'x' else 2
                if block.shape != (size, expected):
                    raise ValueError('invalid physical block dimension')
                if term['kind'] == 'x':
                    r, c = np.nonzero(block)
                    ri.extend(offset+r); ci.extend(pc[term['index']]+c)
                    entries.extend(block[r, c])
                elif term['kind'] == 'l':
                    start = columns[term['index']]
                    ab[offset:offset+size, start:start+2] = block
                else:
                    raise ValueError('unknown physical variable')
            offset += size
        ap = sparse.csc_matrix((entries, (ri, ci)), shape=(rows, 3*len(poses)))
        reduced = eliminate(ap, ab, rhs)
        local_models.append((robot, reduced))
        h += reduced.h; g += reduced.g
    model = normalize(h, g, owners)
    optimum = np.linalg.solve(h, -g)
    for _, reduced in local_models:
        full_residual = max(full_residual, reduced.private_residual(optimum))
    return model, dict(robots=robots, objects=len(labels), dimension=len(g),
        local_private_max_relative_residual=full_residual,
        scope='Frozen physical Jacobian, exact root column already excluded; nominal weights unchanged',
        row_setup='Offline pre-distributed owner rows, not a model-exchange implementation')


def pose3_factors(data):
    """Pinned 2-robot, 24-pose, 4-object SE3 calibration fixture layout."""
    if data.get('schema') != 'dranppr_pose3_physical_calibration_v1' or data.get('truth_used_in_fit'):
        raise ValueError('unsupported SE3 fixture')
    a, rhs = np.asarray(data['a'], float), np.asarray(data['b'], float)
    if a.shape != (708, 162) or rhs.shape != (708,):
        raise ValueError('unexpected pinned SE3 fixture layout')
    # Existing exporter sorts L(0..3), X(1..23), omitting exactly X(0).
    # Each robot emits 11 odometry and 48 object-pose factors, six rows each.
    locals_ = [eliminate(a[:354, 24:90], a[:354, :24], rhs[:354]),
               eliminate(a[354:, 90:], a[354:, :24], rhs[354:])]
    if np.any(a[:354, 90:]) or np.any(a[354:, 24:90]):
        raise ValueError('private SE3 trajectory partition changed')
    h, g = sum(r.h for r in locals_), sum(r.g for r in locals_)
    model = normalize(h, g, [i % 2 for i in range(4) for _ in range(6)])
    optimum = np.linalg.solve(h, -g)
    recovered = np.r_[optimum, locals_[0].recover(optimum), locals_[1].recover(optimum)]
    reference = np.linalg.lstsq(a, rhs, rcond=None)[0]
    error = np.linalg.norm(recovered-reference)/max(np.linalg.norm(reference), np.finfo(float).tiny)
    return model, dict(robots=[0, 1], objects=4, dimension=24,
        full_solve_relative_error=float(error),
        local_private_max_relative_residual=max(r.private_residual(optimum) for r in locals_),
        scope='Genuine SE3 Jacobian with 6-DoF object poses; fixed tangent quadratic ONLY',
        row_setup='Offline pre-distributed owner rows, not a model-exchange implementation')
