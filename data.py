from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "EMA12", "EMA26", "MACD", "MACD_SIGNAL", "RSI14"]

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===== EMA =====
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # ===== MACD =====
    df["MACD"] = df["EMA12"] - df["EMA26"]

    # Signal line
    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # Histogram
    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    df = df.dropna().copy()

    return df
@dataclass
class RegressionDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    raw_df: pd.DataFrame
    sequence_index: Dict[str, pd.DatetimeIndex]
    close_at_t: Dict[str, np.ndarray]
    target_close: Dict[str, np.ndarray]
    target: str
    lookback: int

def resolve_end_date(end: Optional[str]) -> str:
    if end:
        return end
    tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")

def download_stock_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    end_date = resolve_end_date(end)

    df = yf.download(
        ticker,
        start=start,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"Khong tai duoc du lieu cho ma {ticker}")

    # flatten multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # ===== THÊM EMA + MACD + RSI =====

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()

    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACD_SIGNAL"] = (
        df["MACD"].ewm(span=9, adjust=False).mean()
    )

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()

    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["RSI14"] = 100 - (100 / (1 + rs))

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Du lieu cua {ticker} thieu cot: {missing}")

    return df[FEATURE_COLUMNS].copy().dropna().sort_index()

def create_supervised_sequences(
    df: pd.DataFrame,
    lookback: int,
    target: str = "price",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    target='price': y = Close[t+1]
    target='return': y = log(Close[t+1] / Close[t])
    """
    if target not in {"price", "return"}:
        raise ValueError("target phai la 'price' hoac 'return'.")

    values = df[FEATURE_COLUMNS].values.astype(np.float32)
    close = df["Close"].values.astype(np.float32)
    dates = df.index

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    idx_list: List[pd.Timestamp] = []
    close_t_list: List[float] = []
    target_close_list: List[float] = []

    for i in range(lookback - 1, len(df) - 1):
        start = i - lookback + 1
        end = i + 1

        close_t = float(close[i])
        close_next = float(close[i + 1])
        y_value = close_next if target == "price" else float(np.log(close_next / close_t))

        X_list.append(values[start:end])
        y_list.append(y_value)
        idx_list.append(dates[i + 1])
        close_t_list.append(close_t)
        target_close_list.append(close_next)

    if not X_list:
        raise ValueError("Khong du du lieu de tao chuoi. Hay giam lookback hoac mo rong khoang thoi gian.")

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32).reshape(-1, 1),
        pd.DatetimeIndex(idx_list),
        np.asarray(close_t_list, dtype=np.float32),
        np.asarray(target_close_list, dtype=np.float32),
    )

def split_by_time(n_samples: int, train_ratio: float, val_ratio: float) -> Tuple[slice, slice, slice]:
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_samples:
        raise ValueError("Tap train/val/test khong hop le. Hay mo rong khoang thoi gian hoac giam lookback.")
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)

def standardize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    feature_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std())
    if target_std < 1e-8:
        target_std = 1.0

    X_train_s = ((X_train - feature_mean) / feature_std).astype(np.float32)
    X_val_s = ((X_val - feature_mean) / feature_std).astype(np.float32)
    X_test_s = ((X_test - feature_mean) / feature_std).astype(np.float32)

    y_train_s = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_s = ((y_val - target_mean) / target_std).astype(np.float32)
    y_test_s = ((y_test - target_mean) / target_std).astype(np.float32)

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, feature_mean.astype(np.float32), feature_std.astype(np.float32), target_mean, target_std

def prepare_regression_data(
    ticker: str,
    start: str,
    end: Optional[str],
    lookback: int,
    target: str,
    train_ratio: float,
    val_ratio: float,
) -> RegressionDataset:
    raw_df = download_stock_data(ticker=ticker, start=start, end=end)
    X, y, idx, close_t, target_close = create_supervised_sequences(raw_df, lookback=lookback, target=target)

    train_s, val_s, test_s = split_by_time(len(X), train_ratio=train_ratio, val_ratio=val_ratio)

    X_train, X_val, X_test, y_train, y_val, y_test, f_mean, f_std, t_mean, t_std = standardize_data(
        X[train_s], X[val_s], X[test_s],
        y[train_s], y[val_s], y[test_s],
    )

    return RegressionDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_mean=f_mean,
        feature_std=f_std,
        target_mean=t_mean,
        target_std=t_std,
        raw_df=raw_df,
        sequence_index={"train": idx[train_s], "val": idx[val_s], "test": idx[test_s]},
        close_at_t={"train": close_t[train_s], "val": close_t[val_s], "test": close_t[test_s]},
        target_close={"train": target_close[train_s], "val": target_close[val_s], "test": target_close[test_s]},
        target=target,
        lookback=lookback,
    )

def inverse_transform_target(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean

def target_to_close(target_values: np.ndarray, close_at_t: np.ndarray, target: str) -> np.ndarray:
    target_values = np.asarray(target_values).reshape(-1)
    close_at_t = np.asarray(close_at_t).reshape(-1)

    if target == "price":
        return target_values

    return close_at_t * np.exp(target_values)

def make_future_dates(
    last_date: pd.Timestamp,
    forecast_start: Optional[str],
    forecast_end: Optional[str],
    steps: int,
    freq: str = "B",
) -> pd.DatetimeIndex:
    if forecast_start and forecast_end:
        return pd.date_range(start=pd.Timestamp(forecast_start), end=pd.Timestamp(forecast_end), freq=freq)

    start = pd.Timestamp(last_date).normalize() + BDay(1)
    return pd.date_range(start=start, periods=steps, freq=freq)
