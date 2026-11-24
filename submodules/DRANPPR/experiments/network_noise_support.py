"""Native linearization adapters for a bounded classical REML diagnostic."""
import copy
import math

import numpy as np
from scipy import sparse


def assemble(data, split=False, sparse_output=False):
    if data['schema'] != 'dranppr_physical_linearization_v1' or data.get('truth_used', False):
        raise ValueError('unsupported or truth-assisted linearization')
    factors = data['factors']
    columns, sizes = {}, {}
    count = 0
    def identity(factor, term):
        return (term['kind'], factor['robot'] if split or term['kind'] == 'x' else -1, term['index'])
    for f in factors:
        for term in f['terms']:
            key = identity(f, term)
            size = {'x':3, 'l':2}[term['kind']]
            if key not in columns:
                columns[key], sizes[key] = count, size
                count += size
    rows = sum(len(f['b']) for f in factors)
    a = None if sparse_output else np.zeros((rows, count))
    b, groups = np.zeros(rows), []
    row_indices, column_indices, entries = [], [], []
    offset = 0
    for f in factors:
        n = len(f['b'])
        if n != {'odometry':3, 'observation':2}[f['kind']]:
            raise ValueError('unexpected physical factor dimension')
        b[offset:offset+n] = f['b']
        groups.extend([0,1,2] if f['kind'] == 'odometry' else [3,4])
        seen = set()
        for term in f['terms']:
            key = identity(f, term)
            block = np.asarray(term['a'])
            if key in seen or block.shape != (n, sizes[key]):
                raise ValueError('invalid/duplicate Jacobian block')
            seen.add(key)
            col = columns[key]
            if sparse_output:
                local_rows, local_columns = np.nonzero(block)
                row_indices.extend(offset+local_rows)
                column_indices.extend(col+local_columns)
                entries.extend(block[local_rows, local_columns])
            else:
                a[offset:offset+n, col:col+sizes[key]] = block
        offset += n
    if sparse_output:
        a = sparse.csr_matrix((entries, (row_indices, column_indices)), shape=(rows, count))
    if not rows or not count or not np.isfinite(a.data if sparse_output else a).all() or not np.isfinite(b).all():
        raise ValueError('empty/nonfinite physical linearization')
    return a, b, np.asarray(groups)


def components(groups, observation_only=False):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not np.issubdtype(groups.dtype, np.integer) or np.any((groups < 0) | (groups > 4)):
        raise ValueError('invalid physical group labels')
    d = np.eye(5)[groups]
    return (d[:, 3:], d[:, :3].sum(axis=1)) if observation_only else (d, np.zeros(len(groups)))


def apply_scales(package, scales, cutoff, added_variance):
    q = np.asarray(scales, dtype=float)
    if (package['schema'] != 'dranppr_frozen_joint_v2' or q.shape != (5,) or
            not np.isfinite(q).all() or np.any(q <= 0) or not math.isfinite(cutoff) or
            not math.isfinite(added_variance) or added_variance < 0):
        raise ValueError('invalid suffix noise specification')
    output = copy.deepcopy(package)
    for r in output['robots']:
        times = {p['index']:p['timestamp_s'] for p in r['poses']}
        for f in r['odometry']:
            if times[f['target']] > cutoff:
                f['sigmas'] = [s*math.sqrt(v*f['dt']) for s,v in zip([.1,.1,.05], q[:3])]
        for f in r['observations']:
            if times[f['pose']] > cutoff:
                f['sigmas'] = [math.sqrt(.05**2*q[3]+added_variance), math.sqrt(.1**2*q[4])]
    output['suffix_variance_multipliers'] = q.tolist()
    output['suffix_added_bearing_variance'] = added_variance
    return output
