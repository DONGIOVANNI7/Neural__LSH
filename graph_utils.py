from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import sys

from enum import StrEnum

class KnnMethods(StrEnum):
    LSH = "lsh"
    HYPERCUBE = "hypercube"
    IVFFLAT = "ivfflat"
    IVFPQ = "ivfpq"

    @staticmethod
    def from_str(input:str):
        match input:
            case "lsh":
                return KnnMethods.LSH
            case "hypercube":
                return KnnMethods.HYPERCUBE
            case "ivfflat":
                return KnnMethods.IVFFLAT
            case "ivfpq":
                return KnnMethods.IVFPQ
            case _:
                raise RuntimeError()



def external_knn(datafile:str, datatype:str, method:KnnMethods, k:int, n:int) -> np.ndarray:
    """
    Run external k-NN approximation program and capture the output
    """

    import subprocess

    proc = subprocess.Popen(
        ["knn/bin/approx_knn", "-d", datafile, "-type", datatype, f"-{method.value}", "-N", str(k)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    knn_indices = np.full((n,k), -1, dtype=np.int64)
    index = 0
    neighbor_pos = 0
    counter = 0

    if proc.stdout == None: raise Exception()

    for next_line in proc.stdout:
        next_line = next_line.strip()

        if next_line == "":
            neighbor_pos = 0
            counter += 1
            if (counter + 1) % 1000 == 0:
                print(f"[kNN] processed {counter+1}/{n} points")

            continue

        val = int(next_line)

        if neighbor_pos == 0:
            index = val
            neighbor_pos = 1
            continue

        knn_indices[index, neighbor_pos-1] = val
        neighbor_pos += 1

    return knn_indices


def brute_force_knn(X: np.ndarray, k: int) -> np.ndarray:
    """
    Implement k-NN graph of X [n, d] without temporary arrays.
    Compute for each i the distance to all other points with:
        dist^2(i,j) = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    Return indices shape [n, k] with k nearest neighbors (without self).
    """
    n = X.shape[0]

    # ||x_i||^2 for each i
    norms = np.einsum("ij,ij->i", X, X)

    knn_indices = np.empty((n, k), dtype=np.int64)

    for i in range(n):
        # dot(x_i, X_j) for each j
        dot = X @ X[i]          # shape (n,)
        dist_sq = norms + norms[i] - 2.0 * dot
        dist_sq[i] = np.inf     # igrnore self

        nn_idx = np.argpartition(dist_sq, k)[:k]
        nn_idx = nn_idx[np.argsort(dist_sq[nn_idx])]
        knn_indices[i] = nn_idx

        if (i + 1) % 1000 == 0:
            print(f"[kNN] processed {i+1}/{n} points")
            sys.stdout.flush()

    return knn_indices


def knn_to_weighted_undirected_edges(
    knn_indices: np.ndarray,
) -> Dict[Tuple[int, int], int]:
    """
    From directed k-NN edges -> undirected with weights:
    - 2 if mutual neighbors
    - 1 if single direction neighor
    Return dict[(i,j)] = weight with i<j.
    """
    n = knn_indices.shape[0]
    edge_weights: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in knn_indices[i]:
            if i == j or j == -1:
                continue
            a, b = (i, j) if i < j else (j, i)
            edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1
    # now each (a,b) has 1 or 2
    return edge_weights


def edges_to_csr(
    n: int, edge_weights: Dict[Tuple[int, int], int]
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Convert undirected weighted edges to CSR for KaHIP.
    Return (vwgt, xadj, adjncy, adjcwgt).
    - vwgt: node weights (1 for all vertices).
    - xadj: prefix sums (len = n+1).
    - adjncy: neighbor indices.
    - adjcwgt: edge weights in same order as adjncy.
    Each edge (i,j) appears in both directions.
    """
    neighbors = [[] for _ in range(n)]
    weights = [[] for _ in range(n)]

    for (i, j), w in edge_weights.items():
        # add both directions
        neighbors[i].append(j)
        weights[i].append(w)
        neighbors[j].append(i)
        weights[j].append(w)

    xadj = [0]
    adjncy: List[int] = []
    adjcwgt: List[int] = []
    for i in range(n):
        xadj.append(xadj[-1] + len(neighbors[i]))
        adjncy.extend(neighbors[i])
        adjcwgt.extend(weights[i])

    vwgt = [1] * n
    return vwgt, xadj, adjncy, adjcwgt

@dataclass
class KaHIPParams:
    m: int
    imbalance: float
    seed: int
    kahip_mode: int

def partition_with_kahip(
    vwgt: List[int],
    xadj: List[int],
    adjncy: List[int],
    adjcwgt: List[int],
    params: KaHIPParams
) -> np.ndarray:
    """
    Run KaHIP (kaffpa) and return labels [n] (0..m-1).
    """
    import kahip

    edgecut, blocks = kahip.kaffpa(
        vwgt,
        xadj,
        adjcwgt,
        adjncy,
        params.m,
        params.imbalance,
        False,  # suppress_output
        params.seed,
        params.kahip_mode,
    )
    return np.asarray(blocks, dtype=np.int64)


def build_partitions(
    knn_indices: np.ndarray,
    params: KaHIPParams
) -> np.ndarray:
    """
    High-level:
    1) Receive k-NN graph
    2) Make graph symmetrical with weights (1/2)
    3) Run through KaHIP
    Return labels [n] (0..m-1).
    """
    print("[KaHIP] Convert knn to weighted undirected graph")
    edge_weights = knn_to_weighted_undirected_edges(knn_indices)
    print("[KaHIP] Prepare graph for partitioning")
    vwgt, xadj, adjncy, adjcwgt = edges_to_csr(knn_indices.shape[0], edge_weights)
    print("[KaHIP] Run partitioning")
    labels = partition_with_kahip(vwgt, xadj, adjncy, adjcwgt, params)
    return labels
