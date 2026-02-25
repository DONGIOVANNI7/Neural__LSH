from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLPClassifier(nn.Module):
    """
    Simple MLP for sorting in m bins.
    net: [Linear -> ReLU] * (layers-1) -> Linear(out=m)
    """

    def __init__(self, d_in: int, n_out: int, n_layers: int = 3, hidden_dim: int = 64):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers = []
        in_features = d_in
        for _ in range(max(n_layers - 1, 0)):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        # output layer
        layers.append(nn.Linear(in_features, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPConfig:
    d_in: int
    n_out: int
    n_layers: int = 3
    hidden_dim: int = 64
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 1


def train_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    config: MLPConfig,
    device: Optional[torch.device] = None,
) -> MLPClassifier:
    """
    Train MLP in (X, y).
    X: [n, d], float32
    y: [n], int64 labels (0..m-1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = MLPClassifier(
        d_in=config.d_in,
        n_out=config.n_out,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"[MLP] Epoch {epoch+1}/{config.epochs} - loss: {avg_loss:.4f}")

    return model


def predict_top_T_bins(
    model: MLPClassifier,
    q: np.ndarray,
    T: int,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    For a query vector q [d] return indices of T bins with greatest softmax probability.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        q_tensor = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).to(device)
        logits = model(q_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # top-T indices (descending)
    if T >= probs.size:
        return np.argsort(-probs)
    top_T = np.argpartition(-probs, T - 1)[:T]
    # sort top_T by probability
    top_T = top_T[np.argsort(-probs[top_T])]
    return top_T
