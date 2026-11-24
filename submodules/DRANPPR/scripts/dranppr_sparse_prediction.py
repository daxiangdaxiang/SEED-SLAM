"""Selected Gaussian covariance from a sparse augmented linear system.

This reuses SuperLU, does not form normal equations, and retains cross terms.
It refactorizes the current model; it is not yet incremental covariance upkeep.
"""
import numpy as np
from scipy.sparse import bmat, csc_matrix, eye
from scipy.sparse.linalg import splu


def sparse_selected_covariance(design, selectors):
    a = csc_matrix(design)
    selectors = np.asarray(selectors, dtype=float)
    if (selectors.ndim != 2 or selectors.shape[0] != a.shape[1] or
            not np.isfinite(a.data).all() or not np.isfinite(selectors).all()):
        raise ValueError('invalid selected covariance system')
    # K [r; dx] = [0; E] implies dx = -(A^T A)^(-1) E.
    matrix = bmat([[eye(a.shape[0], format='csc'), a], [a.T, None]], format='csc')
    rhs = np.vstack([np.zeros((a.shape[0], selectors.shape[1])), selectors])
    try:
        factor = splu(matrix)
        solution = factor.solve(rhs)
    except RuntimeError as error:
        raise ValueError('singular selected covariance model') from error
    covariance = -selectors.T @ solution[a.shape[0]:]
    if not np.isfinite(covariance).all():
        raise ValueError('non-finite selected covariance')
    return .5 * (covariance + covariance.T)
