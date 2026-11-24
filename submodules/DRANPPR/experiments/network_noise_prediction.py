"""Past-only block forecasts with native Jacobians and plug-in noise parameters."""
import math

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse import coo_matrix


def sparse_model(data):
    if data['schema'] != 'dranppr_physical_linearization_v1' or data.get('truth_used', False):
        raise ValueError('invalid native linearization')
    columns, row_ids, col_ids, entries, rhs = {}, [], [], [], []
    size = 0
    for factor in data['factors']:
        residual = np.asarray(factor['b'], dtype=float)
        n = {'odometry': 3, 'observation': 2}[factor['kind']]
        if residual.shape != (n,):
            raise ValueError('invalid native residual')
        offset = len(rhs)
        seen = set()
        for term in factor['terms']:
            key = (term['kind'], factor['robot'] if term['kind'] == 'x' else -1, term['index'])
            dim = {'x': 3, 'l': 2}[term['kind']]
            if key not in columns:
                columns[key] = list(range(size, size+dim))
                size += dim
            block = np.asarray(term['a'], dtype=float)
            if key in seen or block.shape != (n, dim):
                raise ValueError('invalid native Jacobian block')
            seen.add(key)
            i, j = np.nonzero(block)
            row_ids.extend((i+offset).tolist())
            col_ids.extend(np.asarray(columns[key])[j].tolist())
            entries.extend(block[i, j].tolist())
        rhs.extend(residual.tolist())
    if not rhs or not size or not np.isfinite(rhs).all() or not np.isfinite(entries).all():
        raise ValueError('empty or nonfinite native model')
    return coo_matrix((entries, (row_ids, col_ids)), shape=(len(rhs), size)).tocsc(), np.array(rhs), columns


def selected_covariance(a, columns, keys):
    indices, selection = [], {}
    for key in keys:
        if key in selection or key not in columns:
            raise ValueError('invalid covariance key')
        selection[key] = slice(len(indices), len(indices)+len(columns[key]))
        indices.extend(columns[key])
    rhs = np.zeros((a.shape[1], len(indices)))
    rhs[indices, np.arange(len(indices))] = 1.
    # Audit-sized square-root solve avoids squaring the Jacobian condition number.
    r = qr(a.toarray(), mode='r', overwrite_a=True, check_finite=False)[0][:a.shape[1]]
    tolerance = max(a.shape)*np.finfo(float).eps*np.max(np.abs(np.diag(r)))
    if r.shape[0] != r.shape[1] or np.min(np.abs(np.diag(r))) <= tolerance:
        raise ValueError('rank-deficient physical information')
    solved = solve_triangular(r.T, rhs, lower=True, check_finite=False)
    residual = float(np.linalg.norm(r.T@solved-rhs)/max(np.linalg.norm(rhs), 1.))
    covariance = solved.T@solved
    if (not np.isfinite(covariance).all() or residual > 1e-8 or
            np.linalg.eigvalsh(covariance).min() <= 0.):
        raise ValueError('unreliable selected covariance; no damping or floor applied')
    return covariance, selection, residual


def polar_model(pose, point):
    x, y, theta = pose
    dx, dy = np.asarray(point)-[x, y]
    r = math.hypot(dx, dy)
    if r <= 0. or not math.isfinite(r):
        raise ValueError('undefined polar prediction')
    c, s = math.cos(theta), math.sin(theta)
    a, b = c*dx+s*dy, -s*dx+c*dy
    bearing = math.atan2(math.sin(math.atan2(dy, dx)-theta), math.cos(math.atan2(dy, dx)-theta))
    hp = np.array([[b/r**2, -a/r**2, -1.], [-a/r, -b/r, 0.]])
    hl = np.array([[-dy/r**2, dx/r**2], [dx/r, dy/r]])
    return np.array([bearing, r]), hp, hl


def propagate(pose, f, q, delta, dt, scales):
    if not math.isfinite(dt) or dt <= 0.:
        raise ValueError('invalid odometry duration')
    dx, dy, angle = delta
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    next_pose = np.array([x+c*dx-s*dy, y+s*dx+c*dy, theta+angle])
    c, s = math.cos(angle), math.sin(angle)
    # Right-tangent transport Ad(delta^-1), matching native Pose2 factors.
    inverse_x, inverse_y = -c*dx-s*dy, s*dx-c*dy
    step = np.array([[c, s, inverse_y], [-s, c, -inverse_x], [0., 0., 1.]])
    process = np.diag(np.square([.1, .1, .05])*np.asarray(scales)*dt)
    return next_pose, step@f, step@q@step.T+process


def forecast(package, states, linear, events, checkpoint, horizon, scales, mean_response=None):
    scales = np.asarray(scales, dtype=float)
    if (package.get('truth_used', False) or scales.shape != (5,) or np.any(scales <= 0.) or
            not np.isfinite(scales).all() or not math.isfinite(checkpoint) or horizon <= 0.):
        raise ValueError('invalid forecast contract')
    a, _, columns = sparse_model(linear)
    if mean_response is not None and mean_response['columns'] != columns:
        raise ValueError('mean response uses a different state ordering')
    pose_rows = {(int(r['robot']), int(r['index'])): np.array([float(r[k]) for k in ['x','y','yaw']])
                 for r in states if r['kind'] == 'pose'}
    point_rows = {int(r['cluster']): np.array([float(r[k]) for k in ['x','y']])
                  for r in states if r['kind'] == 'landmark'}
    last, known, keys = {}, {}, []
    for r in package['robots']:
        robot = r['id']
        final = max(r['poses'], key=lambda p: p['timestamp_s'])
        if final['timestamp_s'] > checkpoint:
            raise ValueError('state contains future support')
        key = ('x', robot, final['index'])
        if key in columns:
            keys.append(key)
        elif key != ('x', package['root_robot'], 0):
            raise ValueError('missing nonroot pose information')
        last[robot] = dict(pose=pose_rows[robot, final['index']], key=key,
                           index=final['index'], time=final['timestamp_s'], f=np.eye(3), q=np.zeros((3,3)))
        for obj in r['objects']:
            known[robot, obj['label']] = obj['cluster']
            key = ('l', -1, obj['cluster'])
            if key not in keys:
                keys.append(key)
    covariance, selection, solve_residual = selected_covariance(a, columns, keys)
    observation = np.diag(np.square([.05,.1])*scales[3:])
    predictions, unknown = [], 0
    previous = checkpoint
    for event in events:
        t = event['timestamp_s']
        if t <= checkpoint:
            continue
        if t > checkpoint+horizon:
            break
        if t < previous or event['available_time_s'] != t or event['support_max_time_s'] > t:
            raise ValueError('invalid future event order/support')
        previous = t
        robot = event['robot']
        current = last[robot]
        if event['kind'] == 'pose':
            if (event['pose_index'] != current['index']+1 or
                    event['from_pose_index'] != current['index']):
                raise ValueError('nonconsecutive forecast odometry')
            current['pose'], current['f'], current['q'] = propagate(
                current['pose'], current['f'], current['q'], event['delta'], event['duration_s'], scales[:3])
            current.update(index=event['pose_index'], time=t)
            continue
        if event['kind'] != 'observation' or event['pose_index'] != current['index'] or t != current['time']:
            raise ValueError('observation outside atomic frame')
        cluster = known.get((robot, event['landmark_subject']))
        if cluster is None:
            unknown += 1
            continue
        mean, hp, hl = polar_model(current['pose'], point_rows[cluster])
        jacobian = np.zeros((2, covariance.shape[0]))
        if current['key'] in selection:
            jacobian[:, selection[current['key']]] = hp@current['f']
        jacobian[:, selection['l', -1, cluster]] = hl
        state_cov = jacobian@covariance@jacobian.T+hp@current['q']@hp.T
        total = state_cov+observation
        correction = None
        if mean_response is not None:
            from experiments.mean_response import predict_response
            full_j = np.zeros((2, a.shape[1]))
            if current['key'] in columns:
                full_j[:, columns[current['key']]] = hp@current['f']
            full_j[:, columns['l', -1, cluster]] = hl
            direct = np.zeros((2, len(mean_response['robots'])))
            direct[1, mean_response['robots'].index(robot)] = 1.
            shift, extra = predict_response(full_j, direct, mean_response)
            mean = mean+shift
            if shift[0] != 0.:
                mean[0] = math.atan2(math.sin(mean[0]), math.cos(mean[0]))
            total = total+extra
            correction = dict(linear_mean_shift=shift.tolist(), mean_covariance_addition=extra.tolist())
        innovation = np.array([math.atan2(math.sin(event['bearing_rad']-mean[0]),
                                        math.cos(event['bearing_rad']-mean[0])), event['range_m']-mean[1]])
        chol = np.linalg.cholesky(total)
        whitened = np.linalg.solve(chol, innovation)
        energy = float(whitened@whitened)
        nll = .5*(energy+2*np.log(np.diag(chol)).sum()+2*np.log(2*np.pi))
        predictions.append(dict(robot=robot, source_row=event['source_row'], timestamp_s=t,
            subject=event['landmark_subject'], horizon_s=t-checkpoint,
            mean=mean.tolist(), innovation=innovation.tolist(), covariance=total.tolist(),
            observation_covariance=observation.tolist(), energy=energy, nll=float(nll)))
        if correction is not None:
            predictions[-1].update(correction)
    return dict(predictions=predictions, unknown_object_observations=unknown,
        covariance_solve_relative_residual=solve_residual, checkpoint_s=checkpoint, horizon_s=horizon,
        covariance_method='dense_square_root_QR_of_sparse_native_Jacobian',
        truth_used=False, scope='Past-only batch state; odometry-only block forecast; no block observation assimilation; conditional plug-in marginal Gaussian likelihood, not independent joint likelihood')
