"""Linear diagnostic for a fixed range-mean family, not a backend estimator."""
import numpy as np
from scipy.linalg import qr, solve_triangular, svd

from experiments.audit_mean_evidence import range_directions
from experiments.network_noise_prediction import sparse_model, polar_model, propagate


def fit_response(a, residual, directions):
    a, b, directions = np.asarray(a, float), np.asarray(residual, float), np.asarray(directions, float)
    if (a.ndim != 2 or b.shape != (len(a),) or directions.ndim != 2 or
            len(directions) != len(a) or not directions.shape[1] or
            not all(np.isfinite(x).all() for x in (a, b, directions))):
        raise ValueError('invalid fixed linear mean model')
    if len(a) <= a.shape[1]:
        raise ValueError('insufficient state residual degrees of freedom')
    q, r = qr(a, mode='economic', check_finite=False)
    tolerance = max(a.shape)*np.finfo(float).eps*np.max(np.abs(np.diag(r)))
    if np.min(np.abs(np.diag(r))) <= tolerance:
        raise ValueError('rank-deficient state information')
    projected = q.T @ directions
    e = directions-q@projected
    u, singular, vt = svd(e, full_matrices=False, check_finite=False)
    tol = max(a.shape+directions.shape)*np.finfo(float).eps*max(1., np.linalg.norm(directions, 2))
    if len(singular) != directions.shape[1] or np.min(singular) <= tol:
        raise ValueError('rank-deficient mean information; no prior applied')
    beta = -vt.T @ ((u.T @ b)/singular)
    inverse_sqrt = vt.T/singular
    covariance = inverse_sqrt@inverse_sqrt.T
    step = -solve_triangular(r, q.T@b, check_finite=False)
    sensitivity = -solve_triangular(r, projected, check_finite=False)
    orthogonality = float(np.linalg.norm(q.T@e)/max(1., np.linalg.norm(directions)))
    if orthogonality > 1e-10:
        raise ValueError('unreliable projected mean directions')
    after = b+a@step+e@beta
    return dict(step=step, sensitivity=sensitivity, mean=beta, covariance=covariance,
        singular_values=singular, orthogonality_relative=orthogonality,
        original_linear_loss=float(.5*b@b), adjusted_linear_loss=float(.5*after@after),
        stationarity_relative=float(np.linalg.norm(np.column_stack([a, directions]).T@after)/
                                    max(1., np.linalg.norm(a)*np.linalg.norm(b))))


def predict_response(jacobian, direct, response):
    j, d = np.asarray(jacobian, float), np.asarray(direct, float)
    sensitivity = d+j@response['sensitivity']
    shift = j@response['step']+sensitivity@response['mean']
    extra = sensitivity@response['covariance']@sensitivity.T
    if not np.isfinite(shift).all() or not np.isfinite(extra).all():
        raise ValueError('nonfinite linear response prediction')
    return shift, extra


def fit_native_response(linear):
    a, b, columns = sparse_model(linear)
    directions, names = range_directions(linear)
    nr = sum(n.startswith('robot_') for n in names)
    result = fit_response(a.toarray(), b, directions[:, :nr])
    result.update(columns=columns, robots=[int(n.split('_')[1]) for n in names[:nr]])
    return result


def retracted_means(package, states, events, checkpoint, horizon, response):
    """Evaluate the same step nonlinearly; no solve or covariance relinearization.

    The pinned Pose2 chart retracts by right-composing Pose2(dx,dy,dtheta).
    Reuse the tested odometry composition instead of an Expmap substitution.
    """
    delta = response['step']+response['sensitivity']@response['mean']
    poses, points = {}, {}
    for row in states:
        if row['kind'] == 'pose':
            key = ('x', int(row['robot']), int(row['index']))
            value = np.array([float(row[k]) for k in ('x', 'y', 'yaw')])
            if key in response['columns']:
                value = propagate(value, np.eye(3), np.zeros((3, 3)),
                    delta[response['columns'][key]], 1., np.zeros(3))[0]
            poses[key[1:]] = value
        else:
            cluster = int(row['cluster'])
            value = np.array([float(row['x']), float(row['y'])])
            points[cluster] = value+delta[response['columns']['l', -1, cluster]]
    current, known = {}, {}
    for robot in package['robots']:
        last = max(robot['poses'], key=lambda p: p['timestamp_s'])
        if last['timestamp_s'] > checkpoint:
            raise ValueError('future state in retracted forecast')
        current[robot['id']] = dict(pose=poses[robot['id'], last['index']], index=last['index'], time=last['timestamp_s'])
        for obj in robot['objects']:
            known[robot['id'], obj['label']] = obj['cluster']
    rows, previous = [], checkpoint
    for event in events:
        t = event['timestamp_s']
        if t <= checkpoint:
            continue
        if t > checkpoint+horizon:
            break
        if t < previous or event['available_time_s'] != t or event['support_max_time_s'] > t:
            raise ValueError('invalid retracted forecast support')
        previous = t
        rid = event['robot']
        c = current[rid]
        if event['kind'] == 'pose':
            if event['pose_index'] != c['index']+1 or event['from_pose_index'] != c['index']:
                raise ValueError('nonconsecutive retracted forecast odometry')
            c['pose'] = propagate(c['pose'], np.eye(3), np.zeros((3, 3)), event['delta'],
                                  event['duration_s'], np.zeros(3))[0]
            c.update(index=event['pose_index'], time=t)
        else:
            if event['kind'] != 'observation' or event['pose_index'] != c['index'] or t != c['time']:
                raise ValueError('invalid retracted forecast frame')
            cluster = known.get((rid, event['landmark_subject']))
            if cluster is not None:
                mean = polar_model(c['pose'], points[cluster])[0]
                mean[1] += response['mean'][response['robots'].index(rid)]
                rows.append(dict(robot=rid, source_row=event['source_row'], subject=event['landmark_subject'],
                                 timestamp_s=t, mean=mean.tolist()))
    return rows
