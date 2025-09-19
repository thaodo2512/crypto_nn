from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    d = df.copy()
    d[col] = pd.to_datetime(d[col], utc=True)
    d = d.set_index(col).sort_index()
    return d


def winsorize_causal(series: pd.Series, window: int, low: float = 0.01, high: float = 0.99) -> pd.Series:
    """Causal winsorization: clamp x_t using quantiles computed on past window [t-window, t-1].

    - Uses a shifted rolling window to avoid leakage of the current bar.
    - Expands until `window` is reached via `min_periods=1`.
    """
    s = series.astype(float)
    s_shift = s.shift(1)
    q_low = s_shift.rolling(window=window, min_periods=1).quantile(low)
    q_high = s_shift.rolling(window=window, min_periods=1).quantile(high)
    return s.clip(lower=q_low, upper=q_high)


def zscore_causal(series: pd.Series, window: int) -> pd.Series:
    """Causal z-score: z_t = (x_{t-1} - mean_past) / std_past.

    - Uses only past values (shifted by 1) to compute both numerator and denominator.
    - Standard deviation uses ddof=0; zeros are replaced by a small epsilon to avoid division by zero.
    """
    s = series.astype(float)
    sp = s.shift(1)
    mean = sp.rolling(window=window, min_periods=1).mean()
    std = sp.rolling(window=window, min_periods=1).std(ddof=0)
    std = std.replace(0.0, np.finfo(float).eps).fillna(np.finfo(float).eps)
    return (sp - mean) / std


def encode_hour_of_week(ts_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """Encode hour-of-week as sine/cosine in [0, 1] periodic domain.

    - hour_of_week = dayofweek * 24 + hour (0..167)
    - sin = sin(2*pi*hour_of_week/168)
    - cos = cos(2*pi*hour_of_week/168)
    """
    how = (ts_index.dayofweek * 24 + ts_index.hour).astype(int)
    angle = 2 * math.pi * (how / 168.0)
    return np.sin(angle), np.cos(angle)


def safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Compute log(num/den) safely, returning 0 where invalid.

    - If denominator <=0 or numerator <=0 → 0
    """
    n = num.astype(float)
    d = den.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((n > 0) & (d > 0), np.log(n / d), 0.0)
    return pd.Series(r, index=num.index, dtype=float)


def safe_log1p(x: pd.Series) -> pd.Series:
    v = x.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = np.log1p(np.maximum(v, 0.0))
    return pd.Series(arr, index=x.index, dtype=float)


def diff_series(x: pd.Series) -> pd.Series:
    return x.astype(float).diff()


@dataclass
class ImputeResult:
    values: pd.Series
    flags: pd.Series


def ffill_with_limit_and_flag(s: pd.Series, limit: int = 3) -> ImputeResult:
    """Forward-fill with a maximum bar limit; emit a 1/0 imputation flag.

    - values: series after limited ffill
    - flags: 1 where the current value is imputed by ffill, else 0
    """
    base = s.copy()
    filled = base.ffill(limit=limit)
    imputed_flag = (filled.notna() & base.isna()).astype(int)
    return ImputeResult(values=filled, flags=imputed_flag)

