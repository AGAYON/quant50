"""
Label construction utilities for Quant50.

This module provides vectorized functions to build forward-looking labels
for machine learning while avoiding information leakage. It supports
multiple label modes: forward return, excess return vs benchmark, cross-
sectional rank, and multinomial classification. It also provides a
`prepare_training_frame` helper to merge features and labels safely.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _validate_bars(
    bars: pd.DataFrame, required: List[str] | None = None
) -> pd.DataFrame:
    """
    Validate and normalize bars input.

    Ensures presence of required columns and returns a copy sorted by
    ['symbol', 'timestamp'] with NaNs dropped in required columns.
    """
    req = set(["timestamp", "symbol", "close"]) | set(required or [])
    missing = req - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ✅ FIX: combine required + extra columns safely
    cols = list(req) + [c for c in bars.columns if c not in req]
    df = bars.loc[:, cols].copy()

    df = df.dropna(subset=list(req))
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def build_labels_forward_return(
    bars: pd.DataFrame, horizon_days: int = 5
) -> pd.DataFrame:
    """
    Compute forward returns per symbol without leakage.

    label_forward_return_h{h} = close[t + h] / close[t] - 1

    Parameters
    ----------
    bars : pd.DataFrame
        Long-format with at least ['timestamp', 'symbol', 'close'].
    horizon_days : int, optional
        Forward horizon in days (default: 5).

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'symbol', f'label_forward_return_h{horizon_days}']
        Rows with insufficient look-ahead are dropped.
    """
    df = _validate_bars(bars)

    future_close = df.groupby("symbol", sort=False)["close"].shift(-horizon_days)
    fwd = (future_close / df["close"]) - 1.0

    out = df[["timestamp", "symbol"]].copy()
    col = f"label_forward_return_h{horizon_days}"
    out[col] = fwd
    out = out.dropna(subset=[col])
    return out


def build_labels_excess_return(
    bars: pd.DataFrame, benchmark_symbol: str = "SPY", horizon_days: int = 5
) -> pd.DataFrame:
    """
    Compute excess forward return vs benchmark on each date.

    label_excess_h{h} = forward_return(symbol) - forward_return(benchmark)

    Parameters
    ----------
    bars : pd.DataFrame
        Long-format with at least ['timestamp', 'symbol', 'close'].
    benchmark_symbol : str, optional
        Symbol used as benchmark (default: 'SPY').
    horizon_days : int, optional
        Forward horizon (default: 5).

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'symbol', f'label_excess_h{horizon_days}']
    """
    df = _validate_bars(bars)

    # Compute forward returns for all symbols
    future_close = df.groupby("symbol", sort=False)["close"].shift(-horizon_days)
    fwd = (future_close / df["close"]) - 1.0

    # Extract benchmark fwd returns aligned by timestamp
    bench_mask = df["symbol"] == benchmark_symbol
    bench = pd.DataFrame(
        {
            "timestamp": df.loc[bench_mask, "timestamp"],
            "bench_fwd": fwd.loc[bench_mask].values,
        }
    ).dropna()
    bench = bench.drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    # Join benchmark on timestamp
    out = df[["timestamp", "symbol"]].copy()
    out = out.join(bench, on="timestamp")
    out["sym_fwd"] = fwd.values
    out["excess"] = out["sym_fwd"] - out["bench_fwd"]

    col = f"label_excess_h{horizon_days}"
    out = out.dropna(subset=["excess"])  # drop rows without either side
    out = out[["timestamp", "symbol"]].assign(**{col: out["excess"].values})
    return out


def build_rank_labels(
    bars: pd.DataFrame, horizon_days: int = 5, pct: float = 0.2
) -> pd.DataFrame:
    """
    Compute cross-sectional percentile ranks of forward returns by date.

    label_rank_h{h} in [0, 1]
    """
    df = _validate_bars(bars)
    future_close = df.groupby("symbol", sort=False)["close"].shift(-horizon_days)
    fwd = (future_close / df["close"]) - 1.0

    out = df[["timestamp", "symbol"]].copy()
    out["fwd"] = fwd.values

    # Drop rows where forward return is NaN (tail rows)
    out = out.dropna(subset=["fwd"])

    # Percentile rank by date
    ranks = out.groupby("timestamp", sort=False)["fwd"].rank(method="average", pct=True)
    col = f"label_rank_h{horizon_days}"
    out[col] = ranks.clip(lower=0.0, upper=1.0).values
    return out[["timestamp", "symbol", col]]


def build_classification_labels(
    bars: pd.DataFrame,
    horizon_days: int = 5,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
) -> pd.DataFrame:
    """
    Multinomial classification per date based on forward return percentiles.

    label_cls_h{h} in {-1, 0, 1}: bottom, middle, top.
    """
    df = _validate_bars(bars)
    future_close = df.groupby("symbol", sort=False)["close"].shift(-horizon_days)
    fwd = (future_close / df["close"]) - 1.0

    tmp = df[["timestamp", "symbol"]].copy()
    tmp["fwd"] = fwd.values
    tmp = tmp.dropna(subset=["fwd"])  # remove tail rows

    # Compute thresholds per date
    q_low = tmp.groupby("timestamp", sort=False)["fwd"].transform(
        lambda s: s.quantile(bottom_pct)
    )
    q_high = tmp.groupby("timestamp", sort=False)["fwd"].transform(
        lambda s: s.quantile(1.0 - top_pct)
    )

    cls = pd.Series(np.zeros(len(tmp), dtype=int), index=tmp.index)
    cls = cls.mask(tmp["fwd"] <= q_low, -1)
    cls = cls.mask(tmp["fwd"] >= q_high, 1)

    col = f"label_cls_h{horizon_days}"
    out = tmp[["timestamp", "symbol"]].copy()
    out[col] = cls.values
    return out


def prepare_training_frame(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features (at time t) with labels (t->t+h) without leakage.

    Requirements:
    - features indexed by or containing columns ['timestamp', 'symbol'].
    - labels must contain ['timestamp', 'symbol', '<label cols>'] built with shift(-h).

    Returns a DataFrame with ['timestamp', 'symbol', <features...>, <labels...>]
    and no NaNs.
    """

    def _ensure_ts_sym_cols(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == [
            "timestamp",
            "symbol",
        ]:
            df_out = df.reset_index()
        else:
            df_out = df.copy()
        missing = {"timestamp", "symbol"} - set(df_out.columns)
        if missing:
            raise ValueError(f"Features/labels missing keys: {sorted(missing)}")
        return df_out

    feats = _ensure_ts_sym_cols(features)
    labs = _ensure_ts_sym_cols(labels)

    merged = pd.merge(feats, labs, on=["timestamp", "symbol"], how="inner")
    merged = merged.dropna()
    return merged
