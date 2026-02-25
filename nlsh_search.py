from arguments import search_args

import time
from typing import Tuple

import numpy as np
import torch

from dataset_parser import load_dataset, R_MNIST_DEFAULT, R_SIFT_DEFAULT
from index_io import load_index
from models import MLPClassifier, predict_top_T_bins
from search_utils import exact_knn, exact_range

def neural_lsh_query(
    q: np.ndarray,
    base: np.ndarray,
    model: MLPClassifier,
    inverted_index,
    N: int,
    T: int,
    R: float,
    do_range: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Neural LSH for a query q:
    - select top-T bins
    - collect candidates
    - run exact k-NN for each candidate
    - if do_range: run range search for each candidate
    Return (approx_idx, approx_dist, approx_range_idx)
    """
    # 1. top-T bins
    top_bins = predict_top_T_bins(model, q, T=T, device=device)
    candidates = set()
    for b in top_bins:
        if b < len(inverted_index):
            for idx in inverted_index[b]:
                candidates.add(int(idx))
    candidates = np.asarray(sorted(candidates), dtype=np.int64)

    if candidates.size == 0:
        # fallback: no candidate -> all of base
        candidates = np.arange(base.shape[0], dtype=np.int64)

    # 2. exact search over restricted set
    sub_base = base[candidates]
    diff = sub_base - q
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    dist = np.sqrt(dist_sq)
    order = np.argsort(dist)
    order = order[:N]
    approx_idx = candidates[order]
    approx_dist = dist[order]

    approx_range_idx = np.array([], dtype=np.int64)
    if do_range:
        mask = dist_sq < (R * R)
        range_idx = candidates[mask]
        range_dist = np.sqrt(dist_sq[mask])
        r_order = np.argsort(range_dist)
        approx_range_idx = range_idx[r_order]

    return approx_idx, approx_dist, approx_range_idx


def main() -> None:
    args = search_args()
    dtype = args.type.lower()
    do_range = args.do_range.lower() == "true"

    # Default R if not given
    if args.R is None:
        if dtype == "mnist":
            R = R_MNIST_DEFAULT
        else:
            R = R_SIFT_DEFAULT
    else:
        R = args.R

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Search] Loading index from {args.i}")
    index = load_index(args.i, device=device)
    model = index.model
    inverted_index = index.inverted_index
    index_dtype = index.dataset_type
    base_shape_from_index = index.base_shape

    if index_dtype != dtype:
        print(
            f"[Warning] Index dataset type ({index_dtype}) != CLI type ({dtype}). "
            f"Proceeding but this may be inconsistent."
        )

    print(f"[Search] Loading data and queries")
    base, queries = load_dataset(args.d, args.q, dtype)
    if queries is None:
        raise ValueError("Query file did not load properly")

    if tuple(base.shape) != tuple(base_shape_from_index):
        print(
            f"[Warning] Loaded base shape {base.shape} "
            f"differs from index shape {base_shape_from_index}"
        )

    N = args.N
    T = args.T
    out_path = args.o

    n_queries = queries.shape[0]
    print(
        f"[Search] Starting Neural LSH search: queries={n_queries}, "
        f"N={N}, T={T}, R={R}, range={do_range}"
    )

    ms_approx_total = 0.0
    ms_exact_total = 0.0
    af_sum = 0.0
    af_count = 0.0
    recall_sum = 0

    with open(out_path, "w") as f:
        f.write("Neural LSH\n")

        for qi in range(n_queries):
            if (qi + 1) % 100 == 0 or qi == 0:
                print(f"[Search] Processed {qi+1}/{n_queries} queries", flush=True)
            q = queries[qi]
            f.write(f"Query: {qi}\n")

            # Approximate (Neural LSH) time
            t0 = time.perf_counter()
            approx_idx, approx_dist, approx_range_idx = neural_lsh_query(
                q, base, model, inverted_index, N=N, T=T, R=R, do_range=do_range, device=device
            )
            ms_approx = (time.perf_counter() - t0) * 1000.0
            ms_approx_total += ms_approx

            # Exact baseline time
            t1 = time.perf_counter()
            exact_idx, exact_dist = exact_knn(base, q, N=N)
            ms_exact = (time.perf_counter() - t1) * 1000.0
            ms_exact_total += ms_exact

            # AF & Recall@N
            if exact_dist[0] == 0.0:
                if approx_dist[0] == 0.0:
                    af = 1.0
                else:
                    af = float("nan")
            else:
                af = float(approx_dist[0]) / float(exact_dist[0])
            if np.isfinite(af):
                af_sum += af
                af_count += 1.0

            found_exact = int(exact_idx[0] in approx_idx)
            recall_sum += found_exact

            # Print N neighbors
            for rank in range(len(approx_idx)):
                idx_a = int(approx_idx[rank])
                d_a = float(approx_dist[rank])
                d_t = float(exact_dist[rank]) if rank < len(exact_dist) else float("nan")
                f.write(f"Nearest neighbor-{rank+1}: {idx_a}\n")
                f.write(f"distanceApproximate: {d_a}\n")
                f.write(f"distanceTrue: {d_t}\n\n")

            # Range part
            if do_range:
                f.write(f"\nR-near neighbors:\n") #f.write(f"\nR-near neighbors: {len(approx_range_idx)}\n")
                for idx_r in approx_range_idx:
                    f.write(f"{int(idx_r)}\n")

            f.write("\n")

        # Metrics
        Q = float(n_queries)
        af_avg = (af_sum / af_count) if af_count > 0.0 else float("nan")
        recall_at_n = recall_sum / Q
        t_approx_avg = ms_approx_total / Q
        t_true_avg = ms_exact_total / Q
        qps = 1000.0 * Q / ms_approx_total if ms_approx_total > 0 else 0.0

        f.write(f"Average AF: {af_avg:.6f}\n")
        f.write(f"Recall@N: {recall_at_n:.6f}\n")
        f.write(f"QPS: {qps:.6f}\n")
        f.write(f"tApproximateAverage: {t_approx_avg:.6f}\n")
        f.write(f"tTrueAverage: {t_true_avg:.6f}\n")

    print(f"[Search] Done. Results written to {out_path}")


if __name__ == "__main__":
    main()
