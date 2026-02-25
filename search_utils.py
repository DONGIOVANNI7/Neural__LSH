from typing import Tuple, List

import numpy as np


def exact_knn(
    base: np.ndarray, q: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exhaustive k-NN search over all of base [n, d] for query q [d].
    Return (indices, distances) sorted.
    """
    diff = base - q
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    dist = np.sqrt(dist_sq)
    idx = np.argsort(dist)
    idx = idx[:N]
    return idx, dist[idx]


def exact_range(
    base: np.ndarray, q: np.ndarray, R: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exhaustive range search over all of base [n, d] για query q [d].
    Return (indices, distances) with distance < R sorted.
    """
    diff = base - q
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    mask = dist_sq < (R * R)
    idx = np.nonzero(mask)[0]
    dists = np.sqrt(dist_sq[idx])
    # sort for stability
    order = np.argsort(dists)
    return idx[order], dists[order]
