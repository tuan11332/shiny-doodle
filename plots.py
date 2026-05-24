from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_losses(history, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
        
    plt.plot(history.get("train_loss", []), label="Train loss")
    plt.plot(history.get("val_loss", []), label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    


def plot_predictions(dates, y_true, y_pred, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(pd.to_datetime(dates), y_true, label="Gia thuc te", linewidth=2)
    plt.plot(pd.to_datetime(dates), y_pred, label="Gia du bao", linewidth=2)
    plt.title(title)
    plt.xlabel("Ngay")
    plt.ylabel("Gia dong cua")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_future_forecast(history_df, future_df, ticker: str, model_name: str, output_path: Path, recent_days: int = 160) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hist = history_df.tail(recent_days).copy()
    hist_dates = pd.to_datetime(hist.index)
    future_dates = pd.to_datetime(future_df["date"])

    plt.figure(figsize=(14, 6))
    plt.plot(hist_dates, hist["Close"].values, label=f"Gia thuc te gan nhat ({min(recent_days, len(hist))} ngay)", linewidth=2)
    plt.plot(future_dates, future_df["predicted_close"].values, marker="o", label="Gia du bao tuong lai", linewidth=2)
    plt.axvline(future_dates.iloc[0], linestyle="--", linewidth=1.5, label="Bat dau du bao")
    plt.title(f"{ticker} - {model_name.upper()} du bao gia tuong lai")
    plt.xlabel("Ngay")
    plt.ylabel("Gia dong cua")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
