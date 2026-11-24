"""Native GTSAM sparse Jacobian export in sorted-key order.

GTSAM's exporter omits entries of magnitude at most 1e-12. Keep this explicit
when comparing with the dense reference, rather than claiming bitwise equality.
"""
import numpy as np
from scipy.sparse import csc_matrix


def sparse_linearize(graph, values):
    keys = sorted(graph.keyVector())
    if not keys:
        raise ValueError('physical graph has no variables')
    zero = values.zeroVectors()
    columns = sum(zero.at(key).size for key in keys)
    rows = sum(graph.at(i).dim() for i in range(graph.size()) if graph.at(i) is not None)
    triplets = graph.linearize(values).sparseJacobian_()
    augmented = csc_matrix((triplets[2], (triplets[0].astype(np.int64) - 1,
                                        triplets[1].astype(np.int64) - 1)),
                           shape=(rows, columns + 1))
    return augmented[:, :columns], -augmented[:, columns].toarray().ravel(), keys
