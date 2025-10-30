import numpy as np
import pandas as pd
import pytest

from app.services.features import compute_momentum
from app.services.label import (
    build_classification_labels,
    build_labels_excess_return,
    build_labels_forward_return,
    build_rank_labels,
    prepare_training_frame,
)


@pytest.fixture(scope="module")
def synthetic_multi_symbol_bars() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 90
    dates = pd.bdate_range("2024-02-01", periods=n, freq="B")
    symbols = ["AAA", "BBB", "SPY"]

    frames = []
    for sym, mu in zip(symbols, [0.0004, 0.0006, 0.0005]):
        rets = rng.normal(loc=mu, scale=0.012, size=n)
        close = 100.0 * np.exp(np.cumsum(rets))
        volume = rng.lognormal(mean=13.2, sigma=0.45, size=n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": sym,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def test_forward_return_no_leakage(synthetic_multi_symbol_bars: pd.DataFrame):
    h = 5
    labels = build_labels_forward_return(synthetic_multi_symbol_bars, horizon_days=h)

    # No leakage: last h rows per symbol must be absent
    tail_counts = synthetic_multi_symbol_bars.groupby("symbol").size() - h
    # After dropna, number of label rows per symbol equals n - h
    counts = labels.groupby("symbol").size()
    for sym, n_sym in tail_counts.items():
        assert counts[sym] == n_sym

    # Values finite
    col = f"label_forward_return_h{h}"
    assert not labels[col].isna().any()
    assert np.isfinite(labels[col]).all()


def test_excess_return_against_benchmark(synthetic_multi_symbol_bars: pd.DataFrame):
    h = 5
    exc = build_labels_excess_return(
        synthetic_multi_symbol_bars, benchmark_symbol="SPY", horizon_days=h
    )
    # Alignment by timestamp and subtraction is defined for
    # all rows with both sides present
    col = f"label_excess_h{h}"
    assert not exc[col].isna().any()
    # Excess returns can be positive or negative; ensure finite
    assert np.isfinite(exc[col]).all()


def test_rank_labels_shape_and_range(synthetic_multi_symbol_bars: pd.DataFrame):
    h = 5
    ranks = build_rank_labels(synthetic_multi_symbol_bars, horizon_days=h)
    col = f"label_rank_h{h}"
    # Range [0,1]
    assert (ranks[col] >= 0).all() and (ranks[col] <= 1).all()
    # No NaNs
    assert not ranks[col].isna().any()
    # Each date should have as many ranks as symbols
    per_date = ranks.groupby("timestamp").size().unique()
    assert len(per_date) == 1 and per_date[0] == 3


def test_classification_labels_distribution(synthetic_multi_symbol_bars: pd.DataFrame):
    h = 5
    top, bottom = 0.2, 0.2
    cls = build_classification_labels(
        synthetic_multi_symbol_bars, horizon_days=h, top_pct=top, bottom_pct=bottom
    )
    col = f"label_cls_h{h}"
    assert set(cls[col].unique()).issubset({-1, 0, 1})
    # Rough distribution across dates: with 3 symbols and pct=0.2 thresholds,
    # we expect at least one top/bottom over several dates.
    value_counts = cls[col].value_counts()
    assert value_counts.get(1, 0) > 0
    assert value_counts.get(-1, 0) > 0


def test_prepare_training_frame_merge(synthetic_multi_symbol_bars: pd.DataFrame):
    h = 5
    # Build a simple momentum feature to simulate X
    feats = synthetic_multi_symbol_bars.copy()
    feats = feats.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    mom = compute_momentum(feats, window_days=20)
    mom_df = mom.reset_index().rename(columns={0: "momentum", "momentum": "momentum"})

    # Build labels Y
    labs = build_labels_forward_return(synthetic_multi_symbol_bars, horizon_days=h)

    merged = prepare_training_frame(mom_df, labs)

    # Must contain both feature and label columns
    assert "momentum" in merged.columns
    assert f"label_forward_return_h{h}" in merged.columns
    # No NaNs after merge
    assert not merged.isna().any().any()
    # Size bounded by the limiting window/horizon
    assert len(merged) > 0
