from arguments import build_args

import numpy as np
import torch

from dataset_parser import load_dataset
from graph_utils import KaHIPParams, KnnMethods, build_partitions, external_knn
from models import MLPConfig, train_mlp_classifier
from index_io import save_index

def main() -> None:
    args = build_args()

    dtype = args.type.lower()
    print(f"[Build] Loading dataset {args.d} (type={dtype})")
    data, _ = load_dataset(args.d, None, dtype)
    n, d = data.shape
    print(f"[Build] Loaded data with shape: {data.shape}")

    # Step 1: k-NN graph
    print(f"[Build] Constructing k-NN graph with k={args.knn}")
    knn_method = KnnMethods.from_str(args.knn_method)
    knn_indices = external_knn(args.d, dtype, knn_method, args.knn, n)

    # Steps 2+3: Convert to weigted graph and run KaHIP
    print(f"[Build] Space partitioning with KaHIP")
    kahip_params = KaHIPParams(
        m=args.m,
        imbalance=args.imbalance,
        kahip_mode=args.kahip_mode,
        seed=args.seed,
    )
    labels = build_partitions(knn_indices, kahip_params)
    if labels.shape[0] != n:
        raise RuntimeError("Partition labels size mismatch with dataset")
    print(f"[Build] KaHIP produced {len(np.unique(labels))} partitions")

    # Step 4: Train MLP sorter
    mlp_cfg = MLPConfig(
        d_in=d,
        n_out=args.m,
        n_layers=args.layers,
        hidden_dim=args.nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print(
        f"[Build] Training MLP: layers={mlp_cfg.n_layers}, nodes={mlp_cfg.hidden_dim}, "
        f"epochs={mlp_cfg.epochs}, batch_size={mlp_cfg.batch_size}, lr={mlp_cfg.lr}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mlp_classifier(data, labels, mlp_cfg, device=device)

    # Inverted index: r -> list of point ids
    inverted_index = []
    for r in range(args.m):
        idx = np.where(labels == r)[0]
        inverted_index.append(idx)
        print(f"[Build] Bin {r}: {len(idx)} points")

    save_index(
        path=args.i,
        dataset_type=dtype,
        base_shape=data.shape,
        knn_k=args.knn,
        kahip_params=kahip_params,
        mlp_config=mlp_cfg,
        model=model,
        partitions=labels,
        inverted_index=inverted_index,
    )


if __name__ == "__main__":
    main()
