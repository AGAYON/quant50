import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.features import generate_features
from app.services.label import build_labels_excess_return, build_labels_forward_return
from app.services.model import (
    TrainingConfig,
    build_training_dataset,
    fit,
    load_model,
    predict_scores,
    preprocess_cross_sectional,
    purged_kfold_splits,
    save_model_artifacts,
)


@pytest.fixture(scope="module")
def synthetic_bars_multi() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 80
    dates = pd.bdate_range("2024-03-01", periods=n, freq="B")
    symbols = ["AAA", "BBB", "SPY"]
    frames = []
    for sym, mu in zip(symbols, [0.0004, 0.0006, 0.0005]):
        rets = rng.normal(loc=mu, scale=0.012, size=n)
        close = 100.0 * np.exp(np.cumsum(rets))
        volume = rng.lognormal(mean=13.0, sigma=0.5, size=n).astype(float)
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
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture()
def features_and_labels(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi)
    labels = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    return feats.reset_index(), labels


def test_dataset_schema(features_and_labels):
    feats, labs = features_and_labels
    merged = build_training_dataset(feats, labs)
    # Keys present
    assert {"timestamp", "symbol"}.issubset(merged.columns)
    # Target standardized
    assert "y_excess" in merged.columns
    # No NaNs
    assert not merged.isna().any().any()


def test_no_duplicates(features_and_labels):
    feats, labs = features_and_labels
    merged = build_training_dataset(feats, labs)
    # Composite key uniqueness
    assert not merged.duplicated(subset=["timestamp", "symbol"]).any()


def test_no_leakage(synthetic_bars_multi: pd.DataFrame):
    # Forward labels drop last horizon rows per symbol; ensure lengths align accordingly
    h = 5
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_forward_return(synthetic_bars_multi, horizon_days=h)
    merged = build_training_dataset(feats, labs)
    # For each symbol, labels should not exist for last h timestamps
    by_sym = synthetic_bars_multi.groupby("symbol").size()
    counts = merged.groupby("symbol").size()
    for sym, n in by_sym.items():
        assert counts[sym] <= n - h


def test_preprocessing_normalization(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    # Attach a dummy target to reuse API
    feats["y_excess"] = 0.0
    Xy, _ = preprocess_cross_sectional(feats)
    # For each date, means close to 0 and std close to 1 for non-constant cols
    feature_cols = [
        c for c in Xy.columns if c not in {"timestamp", "symbol", "y_excess"}
    ]
    grouped = Xy.groupby("timestamp", sort=False)[feature_cols]
    means = grouped.mean().abs().max().max()
    stds = (grouped.std().replace(0.0, 1.0) - 1.0).abs().max().max()
    assert means < 1e-6
    assert stds < 1e-6


def test_no_nan_after_preprocessing(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    feats["y_excess"] = 0.0
    Xy, _ = preprocess_cross_sectional(feats)
    assert not Xy.isna().any().any()


def test_purged_split(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    df = build_training_dataset(feats, labs)
    splits = purged_kfold_splits(df, n_splits=3, gap_days=5)
    assert len(splits) >= 1
    ts = pd.to_datetime(df["timestamp"]).dt.normalize().values
    for train_idx, valid_idx in splits:
        assert len(set(train_idx) & set(valid_idx)) == 0
        vdates = np.unique(ts[valid_idx])
        purge_start = vdates.min() - np.timedelta64(5, 'D')
        purge_end = vdates.max() + np.timedelta64(5, 'D')
        assert not (
            ((ts[train_idx] >= purge_start) & (ts[train_idx] <= purge_end)).any()
        )


def test_cv_metrics_thresholds(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    train = build_training_dataset(feats, labs)
    cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=999.0)
    bundle = fit(train, config=cfg)
    m = bundle["metrics"]  # type: ignore[index]
    assert {"ic_mean", "r2_mean", "mse_mean", "cv_splits"}.issubset(m.keys())


def test_overfit_guard(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    train = build_training_dataset(feats, labs)
    # Force overfit guard by setting tiny max_ic_gap
    cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=0.0)
    with pytest.raises(ValueError):
        fit(train, config=cfg)


def test_artifacts_exist(tmp_path: Path, synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    train = build_training_dataset(feats, labs)
    cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=999.0)
    bundle = fit(train, config=cfg)
    paths = save_model_artifacts(bundle, str(tmp_path))
    assert Path(paths["model"]).exists()
    assert Path(paths["scaler"]).exists()
    meta_path = Path(paths["meta"])
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert {
        "created_at",
        "model_type",
        "feature_columns",
        "metrics",
        "config",
    }.issubset(meta.keys())


def test_model_meta_schema(tmp_path: Path, synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    train = build_training_dataset(feats, labs)
    cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=999.0)
    bundle = fit(train, config=cfg)
    paths = save_model_artifacts(bundle, str(tmp_path))
    loaded = load_model(paths["model"])
    assert "feature_columns" in loaded
    assert "metrics" in loaded


def test_public_signatures():
    # Sanity: functions imported above exist
    assert callable(build_training_dataset)
    assert callable(preprocess_cross_sectional)
    assert callable(purged_kfold_splits)
    assert callable(fit)
    assert callable(save_model_artifacts)
    assert callable(predict_scores)
    assert callable(load_model)


def test_predict_output_schema(synthetic_bars_multi: pd.DataFrame):
    feats = generate_features(synthetic_bars_multi).reset_index()
    labs = build_labels_excess_return(
        synthetic_bars_multi, benchmark_symbol="SPY", horizon_days=5
    )
    train = build_training_dataset(feats, labs)
    cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=999.0)
    bundle = fit(train, config=cfg)
    scores = predict_scores(bundle, feats)
    assert {"timestamp", "symbol", "score"}.issubset(scores.columns)
    assert len(scores) == len(feats)


def test_weekly_train_cli_smoke(tmp_path: Path, monkeypatch):
    from jobs import weekly_train as wt

    # Run with no data (no DuckDB) to ensure it exits gracefully
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    # Ensure loader returns empty DF
    monkeypatch.setattr(wt, "_load_bars_from_duckdb", lambda: pd.DataFrame())
    # Capture output
    import sys

    argv_bak = sys.argv[:]
    sys.argv = ["weekly_train.py", "--dry-run"]
    try:
        rc = wt.main()
        assert rc == 0
    finally:
        sys.argv = argv_bak


def test_train_report_exists(
    tmp_path: Path, monkeypatch, synthetic_bars_multi: pd.DataFrame
):
    from jobs import weekly_train as wt

    # Patch loader to return synthetic data and redirect outputs to tmp
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wt, "_load_bars_from_duckdb", lambda: synthetic_bars_multi)
    monkeypatch.setattr(wt, "REPORTS_DIR", str(tmp_path / "reports"))
    import sys

    argv_bak = sys.argv[:]
    sys.argv = [
        "weekly_train.py",
        "--allow-weak",
    ]  # not dry-run, will write artifacts and report
    try:
        rc = wt.main()
        assert rc == 0
        # Verify report
        reports = list((tmp_path / "reports").glob("train_report_*.json"))
        assert len(reports) >= 1
    finally:
        sys.argv = argv_bak
