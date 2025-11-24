import numpy as np
import pandas as pd
import pytest

from app.services.features import (
    compute_liquidity,
    compute_momentum,
    compute_rsi,
    compute_volatility,
    generate_features,
)


@pytest.fixture(scope="module")
def synthetic_bars_df() -> pd.DataFrame:
    """
    Build a reproducible long-format OHLCV-like DataFrame for a single symbol
    over 100 business days with smooth dynamics to avoid numerical pathologies.
    Only 'timestamp', 'symbol', 'close', 'volume' are required by the API.
    """
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")

    # Random-walk prices with small noise around 0 mean
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))

    # Volumes: positive, around 1e6 with moderate variability
    volume = rng.lognormal(mean=13.5, sigma=0.4, size=n).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["TEST"] * n,
            "close": close,
            "volume": volume,
        }
    )
    return df


def test_compute_momentum_basic(synthetic_bars_df: pd.DataFrame):
    window = 20
    s = compute_momentum(synthetic_bars_df, window_days=window)
    s_clean = s.dropna()

    # Tamaño tras ventana
    assert len(s_clean) == len(synthetic_bars_df) - window

    # No NaNs y valores finitos
    assert not s_clean.isna().any()
    assert np.isfinite(s_clean.values).all()

    # Índices esperados
    assert isinstance(s_clean.index, pd.MultiIndex)
    assert list(s_clean.index.names) == ["timestamp", "symbol"]


def test_compute_volatility_basic(synthetic_bars_df: pd.DataFrame):
    window = 20
    s = compute_volatility(synthetic_bars_df, window_days=window)
    s_clean = s.dropna()

    # Tamaño tras ventana
    assert len(s_clean) == len(synthetic_bars_df) - window

    # No NaNs y rango plausible (>0)
    assert not s_clean.isna().any()
    assert (s_clean > 0).all()


def test_compute_rsi_basic(synthetic_bars_df: pd.DataFrame):
    window = 14
    s = compute_rsi(synthetic_bars_df, window_days=window)
    s_clean = s.dropna()

    # Tamaño tras primeros NaNs de arranque
    assert len(s_clean) >= len(synthetic_bars_df) - window

    # RSI en [0, 100]
    assert not s_clean.isna().any()
    assert (s_clean >= 0).all() and (s_clean <= 100).all()


def test_compute_liquidity_basic(synthetic_bars_df: pd.DataFrame):
    window = 20
    s = compute_liquidity(synthetic_bars_df, window_days=window)
    s_clean = s.dropna()

    # Tamaño tras ventana
    assert len(s_clean) == len(synthetic_bars_df) - window

    # No NaNs y rango plausible (>0)
    assert not s_clean.isna().any()
    assert (s_clean > 0).all()


def test_generate_features_columns_and_shapes(synthetic_bars_df: pd.DataFrame):
    feats = generate_features(synthetic_bars_df)

    # Columnas requeridas
    expected_cols = ["momentum", "volatility", "rsi", "liquidity"]
    for c in expected_cols:
        assert c in feats.columns

    # Sin NaNs después de la combinación y limpieza
    assert not feats.isna().any().any()

    # Tamaño: limitado por la ventana más grande (20)
    n = len(synthetic_bars_df)
    assert len(feats) == n - 20
