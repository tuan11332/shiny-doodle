from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.data import (
    FEATURE_COLUMNS,
    inverse_transform_target,
    make_future_dates,
    prepare_regression_data,
    target_to_close,
)
from src.models import GRURegressor, LSTMRegressor, TCNRegressor
from src.plots import plot_future_forecast, plot_losses, plot_predictions
from src.training import compute_metrics, make_dataloaders, predict, set_seed, train_regressor


MODEL_FACTORY = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "tcn": TCNRegressor,
}


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def forecast_future(model, dataset, device, forecast_start, forecast_end, forecast_steps, forecast_freq) -> pd.DataFrame:
    raw_features = dataset.raw_df[FEATURE_COLUMNS].copy()
    last_real_date = raw_features.index[-1]
    last_real_close = float(raw_features["Close"].iloc[-1])

    future_dates = make_future_dates(last_real_date, forecast_start, forecast_end, forecast_steps, forecast_freq)
    if len(future_dates) == 0:
        raise ValueError("Khong tao duoc ngay du bao. Hay kiem tra forecast_start/forecast_end.")

    window = raw_features.tail(dataset.lookback).values.astype(np.float32)

    close_idx = FEATURE_COLUMNS.index("Close")
    open_idx = FEATURE_COLUMNS.index("Open")
    high_idx = FEATURE_COLUMNS.index("High")
    low_idx = FEATURE_COLUMNS.index("Low")
    volume_idx = FEATURE_COLUMNS.index("Volume")

    rows = []

    for step, date in enumerate(future_dates, start=1):
        X_scaled = ((window - dataset.feature_mean) / dataset.feature_std).astype(np.float32)
        pred_scaled = predict(model, X_scaled[np.newaxis, :, :], device=device)
        pred_target = float(inverse_transform_target(pred_scaled, dataset.target_mean, dataset.target_std).flatten()[0])

        prev_close = float(window[-1, close_idx])
        pred_close_raw = pred_target if dataset.target == "price" else float(prev_close * np.exp(pred_target))

        # Chặn biến động quá lớn khi forecast đệ quy để tránh giá âm hoặc spike vô lý.
        # Giới hạn biến động tối đa mỗi ngày (2%)
        max_move = 0.02

        # Chặn spike vô lý
        if abs(pred_close_raw - prev_close) / prev_close > max_move:
            pred_close = prev_close * (
                1 + np.sign(pred_close_raw - prev_close) * max_move
            )
        else:
            pred_close = pred_close_raw

        # Smooth forecast để dự báo mượt hơn
        pred_close = 0.7 * prev_close + 0.3 * pred_close

        # Không cho giá âm
        pred_close = max(pred_close, 0.01)

        prev_row = window[-1].copy()
        ratio = pred_close / prev_close if prev_close > 0 else 1.0

        new_row = prev_row.copy()
        new_row[open_idx] = prev_row[open_idx] * ratio
        new_row[high_idx] = max(prev_row[high_idx] * ratio, pred_close)
        new_row[low_idx] = min(prev_row[low_idx] * ratio, pred_close)
        new_row[close_idx] = pred_close
        new_row[volume_idx] = float(np.mean(window[-20:, volume_idx]))

        rows.append(
            {
                "step": step,
                "date": date.date(),
                "predicted_close": pred_close,
                "last_real_close": last_real_close,
                "change_from_latest_%": ((pred_close - last_real_close) / last_real_close) * 100 if last_real_close else np.nan,
            }
        )

        window = np.vstack([window[1:], new_row]).astype(np.float32)

    return pd.DataFrame(rows)


def run_one(ticker: str, model_name: str, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    print(f"\n===== {ticker} | {model_name.upper()} | target={args.target} =====")

    dataset = prepare_regression_data(
        ticker=ticker,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        target=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    out_dir = Path(args.output_dir) / ticker / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
        args.batch_size,
    )

    model = MODEL_FACTORY[model_name](
        input_size=dataset.X_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model_path = out_dir / "model.pt"

    # ===== LOAD MODEL =====
    if model_path.exists():

        print(f"Loading trained model from: {model_path}")

        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        model.to(device)
        model.eval()

        from types import SimpleNamespace
        import json

        history_path = out_dir / "history.json"

        if history_path.exists():

            with open(history_path, "r") as f:
                loaded_history = json.load(f)

        else:

            loaded_history = {
                "train_loss": [],
                "val_loss": [],
            }

        train_result = SimpleNamespace(
            best_val_loss=0.0,
            history=loaded_history
        )

    # ===== TRAIN MODEL =====
    else:

        print("Training new model...")

        train_result = train_regressor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
            patience=args.patience,
            loss_name=args.loss,
        )

        torch.save(model.state_dict(), model_path)

        import json

        history_path = out_dir / "history.json"

        with open(history_path, "w") as f:
            json.dump(train_result.history, f)

        print(f"Saved model to: {model_path}")

    pred_test_scaled = predict(model, dataset.X_test, device=device)
    pred_target = inverse_transform_target(pred_test_scaled, dataset.target_mean, dataset.target_std)
    true_target = inverse_transform_target(dataset.y_test, dataset.target_mean, dataset.target_std)

    y_pred_close = target_to_close(pred_target, dataset.close_at_t["test"], dataset.target)
    y_true_close = dataset.target_close["test"]

    metrics = compute_metrics(y_true_close, y_pred_close)

    naive_pred_close = dataset.close_at_t["test"]
    naive_metrics = {f"Naive_{k}": v for k, v in compute_metrics(y_true_close, naive_pred_close).items()}

    metrics.update(naive_metrics)
    metrics.update(
        {
            "ticker": ticker,
            "model": model_name.upper(),
            "target": args.target,
            "lookback": args.lookback,
            "loss": args.loss.upper(),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "best_val_loss": float(train_result.best_val_loss),
            "train_sequences": int(len(dataset.X_train)),
            "val_sequences": int(len(dataset.X_val)),
            "test_sequences": int(len(dataset.X_test)),
            "test_start": str(dataset.sequence_index["test"][0].date()),
            "test_end": str(dataset.sequence_index["test"][-1].date()),
            "last_real_date": str(dataset.raw_df.index[-1].date()),
            "last_real_close": float(dataset.raw_df["Close"].iloc[-1]),
        }
    )

    pred_df = pd.DataFrame(
        {
            "date": dataset.sequence_index["test"],
            "close_t": dataset.close_at_t["test"],
            "y_true_close": y_true_close,
            "y_pred_close": y_pred_close,
            "abs_error": np.abs(y_true_close - y_pred_close),
            "true_target": true_target.flatten(),
            "pred_target": pred_target.flatten(),
            "naive_pred_close": naive_pred_close,
        }
    )
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    future_df = forecast_future(
        model=model,
        dataset=dataset,
        device=device,
        forecast_start=args.forecast_start,
        forecast_end=args.forecast_end,
        forecast_steps=args.forecast_steps,
        forecast_freq=args.forecast_freq,
    )
    future_df.to_csv(out_dir / "future_forecast.csv", index=False)

    if len(future_df) > 0:
        metrics["forecast_start"] = str(future_df["date"].iloc[0])
        metrics["forecast_end"] = str(future_df["date"].iloc[-1])
        metrics["forecast_last_close"] = float(future_df["predicted_close"].iloc[-1])
        metrics["forecast_total_change_%"] = float(future_df["change_from_latest_%"].iloc[-1])

    save_json(metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")

    plot_predictions(
        dataset.sequence_index["test"],
        y_true_close,
        y_pred_close,
        f"{ticker} - {model_name.upper()} du bao vs thuc te",
        out_dir / "prediction_vs_actual.png",
    )
    plot_losses(train_result.history, f"{ticker} - {model_name.upper()} training loss", out_dir / "loss_curve.png")
    plot_future_forecast(dataset.raw_df, future_df, ticker, model_name, out_dir / "future_forecast.png", args.recent_days_plot)

    print(f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.4f}%")
    print(f"Da luu ket qua tai: {out_dir}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Do an #28 - Du bao gia co phieu bang LSTM/GRU/TCN regression")

    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)

    parser.add_argument("--target", choices=["price", "return"], default="price")
    parser.add_argument("--lookback", type=int, default=60)

    parser.add_argument("--forecast_start", type=str, default=None)
    parser.add_argument("--forecast_end", type=str, default=None)
    parser.add_argument("--forecast_steps", type=int, default=20)
    parser.add_argument("--forecast_freq", choices=["B", "D"], default="B")

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--models", nargs="+", choices=["lstm", "gru", "tcn"], default=["lstm", "gru"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--loss", choices=["mse", "mae"], default="mse")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--output_dir", default="results_price_regression")
    parser.add_argument("--recent_days_plot", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung device: {device}")

    all_metrics: List[Dict[str, Any]] = []

    for ticker in args.tickers:
        for model_name in args.models:
            all_metrics.append(run_one(ticker, model_name, args, device))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "all_metrics.csv", index=False)

    print("\n===== Tong hop ket qua =====")
    display_cols = [
        "ticker",
        "model",
        "target",
        "loss",
        "MAE",
        "RMSE",
        "MAPE",
        "DirectionAccuracy_%",
        "Naive_MAE",
        "Naive_RMSE",
        "forecast_start",
        "forecast_end",
        "forecast_last_close",
    ]
    existing = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[existing].to_string(index=False))

    print(f"\nDa luu tong hop tai: {out_dir / 'all_metrics.csv'}")


if __name__ == "__main__":
    main()
