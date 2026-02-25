from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch

from graph_utils import KaHIPParams
from models import MLPConfig, MLPClassifier

def save_index(
    path: str,
    dataset_type: str,
    base_shape: tuple,
    knn_k: int,
    kahip_params: KaHIPParams,
    mlp_config: MLPConfig,
    model: MLPClassifier,
    partitions: np.ndarray,
    inverted_index: List[np.ndarray],
) -> None:
    """
    Store all the information of the index in a torch .pt file.
    """
    state = {
        "dataset_type": dataset_type,
        "base_shape": base_shape,
        "knn_k": knn_k,
        "kahip_params": kahip_params.__dict__,
        "mlp_config": mlp_config.__dict__,
        "model_state_dict": model.state_dict(),
        "partitions": partitions.astype(np.int64),
        "inverted_index": [arr.astype(np.int64).tolist() for arr in inverted_index],
    }
    torch.save(state, path)
    print(f"[Index] Saved index to {path}")


@dataclass
class Index:
    dataset_type: str
    base_shape: tuple
    knn_k: int
    kahip_params: KaHIPParams
    mlp_config: MLPConfig
    model: MLPClassifier
    partitions: np.ndarray
    inverted_index: List[np.ndarray]


def load_index(
    path: str, device: torch.device
) -> Index:
    """
    Load index and rebuild MLP model.
    Return Index:
    """
    state = torch.load(path, map_location=device, weights_only=False)
    dataset_type = state["dataset_type"]
    base_shape = tuple(state["base_shape"])
    knn_k = int(state["knn_k"])
    kahip_params = KaHIPParams(**state["kahip_params"])
    mlp_cfg = MLPConfig(**state["mlp_config"])

    model = MLPClassifier(
        d_in=mlp_cfg.d_in,
        n_out=mlp_cfg.n_out,
        n_layers=mlp_cfg.n_layers,
        hidden_dim=mlp_cfg.hidden_dim,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    partitions = np.asarray(state["partitions"], dtype=np.int64)
    inverted_index = [
        np.asarray(lst, dtype=np.int64) for lst in state["inverted_index"]
    ]

    return Index(
        dataset_type=dataset_type,
        base_shape=base_shape,
        knn_k=knn_k,
        kahip_params=kahip_params,
        mlp_config=mlp_cfg,
        model=model,
        partitions=partitions,
        inverted_index=inverted_index,
    )
