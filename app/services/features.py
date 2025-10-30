"""
Feature engineering services for Quant50.

This module defines the public API for quantitative feature computation
aligned with the PRD (momentum, volatility, RSI, liquidity) and a
`generate_features` orchestrator that builds a feature matrix for the
universe stored in DuckDB.

Notes
-----
- The concrete implementations must follow the PRD constraints and
  compute cross-sectional features on daily bars.
- Keep functions deterministic and side-effect free. IO should be
  performed outside these functions (e.g., in service callers).
"""

import numpy as np
import pandas as pd


def compute_momentum(
    prices: pd.DataFrame,
    window_days: int = 20,
) -> pd.Series:
    """
    Compute price momentum over a rolling window.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide or long format daily close prices. Expected at minimum the
        columns ['timestamp', 'symbol', 'close'] if long format.
    window_days : int, optional
        Rolling window in days, by default 20.

    Returns
    -------
    pd.Series
        Momentum signal aligned with input index (per symbol and date).
    """
    required_cols = {"timestamp", "symbol", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = (
        prices[["timestamp", "symbol", "close"]]
        .dropna()
        .sort_values(["symbol", "timestamp"])
        .copy()
    )

    close_shifted = df.groupby("symbol", sort=False)["close"].shift(window_days)
    momentum = (df["close"] / close_shifted) - 1.0

    # Build an index aligned Series on (timestamp, symbol)
    idx = pd.MultiIndex.from_frame(df[["timestamp", "symbol"]])
    momentum.name = "momentum"
    momentum.index = idx
    return momentum


def compute_volatility(
    prices: pd.DataFrame,
    window_days: int = 20,
) -> pd.Series:
    """
    Compute realized volatility over a rolling window.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data; expects ['timestamp', 'symbol', 'close'] in long format
        or a wide matrix of prices.
    window_days : int, optional
        Rolling window length, by default 20.
    annualization_factor : int, optional
        Trading days per year used to annualize volatility, by default 252.

    Returns
    -------
    pd.Series
        Annualized volatility per symbol-date.
    """
    required_cols = {"timestamp", "symbol", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = (
        prices[["timestamp", "symbol", "close"]]
        .dropna()
        .sort_values(["symbol", "timestamp"])
        .copy()
    )

    log_price = np.log(df["close"])  # type: ignore[arg-type]
    log_ret = log_price.groupby(df["symbol"], sort=False).diff()
    vol = (
        log_ret.groupby(df["symbol"], sort=False)
        .rolling(window_days)
        .std()
        .reset_index(level=0, drop=True)
    )

    idx = pd.MultiIndex.from_frame(df[["timestamp", "symbol"]])
    vol.name = "volatility"
    vol.index = idx
    return vol


def compute_rsi(
    prices: pd.DataFrame,
    window_days: int = 14,
) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) over a rolling window.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data; expects ['timestamp', 'symbol', 'close'] in long format
        or a wide matrix of prices.
    window_days : int, optional
        RSI window length, by default 14.

    Returns
    -------
    pd.Series
        RSI values per symbol-date on a 0-100 scale.
    """
    required_cols = {"timestamp", "symbol", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = (
        prices[["timestamp", "symbol", "close"]]
        .dropna()
        .sort_values(["symbol", "timestamp"])
        .copy()
    )

    delta = df.groupby("symbol", sort=False)["close"].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder's smoothing uses EMA with alpha=1/window
    alpha = 1.0 / float(window_days)
    avg_gain = (
        gains.groupby(df["symbol"], sort=False)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    avg_loss = (
        losses.groupby(df["symbol"], sort=False)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(100.0)  # when avg_loss == 0 → RSI = 100

    idx = pd.MultiIndex.from_frame(df[["timestamp", "symbol"]])
    rsi.name = "rsi"
    rsi.index = idx
    return rsi


def compute_liquidity(
    bars: pd.DataFrame,
    window_days: int = 20,
) -> pd.Series:
    """
    Compute a simple liquidity proxy (e.g., rolling ADV or dollar volume).

    Parameters
    ----------
    bars : pd.DataFrame
        Daily OHLCV data; expects ['timestamp', 'symbol', 'close', 'volume'].
    window_days : int, optional
        Rolling window length, by default 20.

    Returns
    -------
    pd.Series
        Liquidity signal per symbol-date.
    """
    required_cols = {"timestamp", "symbol", "volume"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = (
        bars[["timestamp", "symbol", "volume"]]
        .dropna()
        .sort_values(["symbol", "timestamp"])
        .copy()
    )

    out = []
    for sym, g in df.groupby("symbol", sort=False):
        liq = (
            g["volume"]
            .rolling(window_days, min_periods=window_days)
            .mean()
            .iloc[window_days:]
        )
        idx = pd.MultiIndex.from_product([g["timestamp"].iloc[window_days:], [sym]])
        liq.index = idx
        out.append(liq)
    return pd.concat(out).rename("liquidity")


def generate_features(
    bars: pd.DataFrame,
    momentum_window: int = 20,
    volatility_window: int = 20,
    rsi_window: int = 14,
    liquidity_window: int = 20,
) -> pd.DataFrame:
    """
    Generate the full feature matrix required by the modeling pipeline.

    Parameters
    ----------
    bars : pd.DataFrame
        Long-format daily OHLCV with columns at least
        ['timestamp', 'symbol', 'close', 'volume'].
    momentum_window : int, optional
        Window for momentum computation, by default 20.
    volatility_window : int, optional
        Window for volatility computation, by default 20.
    rsi_window : int, optional
        Window for RSI computation, by default 14.
    liquidity_window : int, optional
        Window for liquidity proxy, by default 20.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by ['timestamp', 'symbol'] with columns
        ['momentum', 'volatility', 'rsi', 'liquidity'].
    """
    required_cols = {"timestamp", "symbol", "close", "volume"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Permitir columnas opcionales (open, high, low)
    optional_cols = [c for c in ["open", "high", "low"] if c in bars.columns]
    df = (
        bars[["timestamp", "symbol", "close", "volume"] + optional_cols]
        .dropna()
        .sort_values(["symbol", "timestamp"])
        .copy()
    )

    mom = compute_momentum(df, window_days=momentum_window)
    vol = compute_volatility(df, window_days=volatility_window)
    rsi = compute_rsi(df, window_days=rsi_window)
    liq = compute_liquidity(df, window_days=liquidity_window)

    # Align and combine all feature series
    features = pd.concat([mom, vol, rsi, liq], axis=1).dropna()
    # Ensure a consistent MultiIndex naming for downstream consumers
    features = features.sort_index()
    try:
        features.index = features.index.set_names(["timestamp", "symbol"])
    except Exception:
        # Fallback: if index is not a MultiIndex, leave as is
        pass
    return features
