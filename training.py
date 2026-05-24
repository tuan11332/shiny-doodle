from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    history: Dict[str, List[float]]
    best_state_dict: Dict[str, torch.Tensor]
    best_val_loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    loss_name: str,
) -> TrainResult:
    criterion = nn.L1Loss() if loss_name.lower() == "mae" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(float(criterion(model(X_batch), y_batch).item()))

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping sau {epoch + 1} epoch.")
            break

    model.load_state_dict(best_state_dict)
    return TrainResult(history, best_state_dict, best_val_loss)


def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []

    with torch.no_grad():
        for (X_batch,) in loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())

    return np.vstack(preds)


def compute_metrics(y_true_close: np.ndarray, y_pred_close: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true_close).reshape(-1)
    y_pred = np.asarray(y_pred_close).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)

    direction_accuracy = float("nan")
    if len(y_true) >= 2:
        direction_accuracy = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE": mape,
        "DirectionAccuracy_%": direction_accuracy,
    }
