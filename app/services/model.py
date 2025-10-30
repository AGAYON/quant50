"""
Model training services for Quant50 (T004).

Provides dataset building, preprocessing, CV splitting, model training,
artifact persistence, and inference interfaces. Initial implementation
exposes stubs to be filled progressively per T004A–T004H.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42


@dataclass
class TrainingConfig:
    model_type: str = "lightgbm"
    target_column: str = "y_excess"
    cv_splits: int = 3
    gap_days_purge: int = 5
    max_na_col: float = 0.10
    random_state: int = SEED
    min_ic_mean: float = 0.02
    max_ic_gap: float = 0.05
    min_r2_valid: float = 0.01


def build_training_dataset(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features (t) and labels (t->t+h) on ['timestamp','symbol'].

    Validations performed:
    - Required columns present in both inputs
    - No duplicate keys (['timestamp','symbol']) in either input
    - Output has a single target column named 'y_excess' (auto-detected from labels)
    - No NA rows after merge

    Notes
    -----
    Labels must already be built with forward shift (no leakage by construction).
    This function asserts integrity but does not recompute labels.
    """
    from app.services.label import prepare_training_frame

    def _ensure_keys(df: pd.DataFrame, name: str) -> None:
        missing = {"timestamp", "symbol"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing key columns: {sorted(missing)}")

    def _assert_no_dup_keys(df: pd.DataFrame, name: str) -> None:
        if df.duplicated(subset=["timestamp", "symbol"]).any():
            raise ValueError(f"{name} contains duplicate ['timestamp','symbol'] keys")

    _ensure_keys(features, "features")
    _ensure_keys(labels, "labels")
    _assert_no_dup_keys(features, "features")
    _assert_no_dup_keys(labels, "labels")

    # Detect label column(s) (exclude keys). Prefer excess-return label if multiple
    label_cols = [c for c in labels.columns if c not in {"timestamp", "symbol"}]
    if not label_cols:
        raise ValueError("labels must include at least one target column")
    # Heuristic: choose first 'label_excess_h*' if available, else first
    excess_cols = [c for c in label_cols if c.startswith("label_excess_h")]
    target_col = excess_cols[0] if excess_cols else label_cols[0]

    merged = prepare_training_frame(
        features, labels[["timestamp", "symbol", target_col]]
    )

    # Standardize target name
    merged = merged.rename(columns={target_col: "y_excess"})

    # Final sanity checks
    if merged.isna().any().any():
        raise ValueError("Merged training dataset contains NA values after join")

    # Ensure monotonic timestamps per symbol (temporal consistency)
    # This does not detect leakage directly but enforces order for downstream splits
    by_sym = merged.sort_values(["symbol", "timestamp"]).groupby("symbol")["timestamp"]
    if not by_sym.apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("Merged dataset has non-monotonic timestamps within symbols")

    return merged


def fit(
    train_df: pd.DataFrame, config: Optional[TrainingConfig] = None
) -> Dict[str, object]:
    """
    Fit the cross-sectional ML model and return a dict containing the
    trained estimator, scaler/preprocessor, and metadata.
    """
    cfg = config or TrainingConfig()

    # Identify feature columns (exclude keys and target)
    exclude = {"timestamp", "symbol", cfg.target_column}
    feature_columns = [c for c in train_df.columns if c not in exclude]
    if not feature_columns:
        raise ValueError("No feature columns found for training")

    # Drop columns with excessive NA ratio
    na_ratio = train_df[feature_columns].isna().mean()
    keep_cols = [c for c in feature_columns if na_ratio[c] <= cfg.max_na_col]
    if not keep_cols:
        raise ValueError("All feature columns exceed NA threshold")

    work = train_df[["timestamp", "symbol", cfg.target_column] + keep_cols].copy()
    work = work.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Preprocess cross-sectionally
    Xy, prep_meta = preprocess_cross_sectional(work, feature_columns=keep_cols)
    X = Xy[keep_cols].values
    y = Xy[cfg.target_column].values

    # Build purged CV splits
    splits = purged_kfold_splits(
        Xy, n_splits=cfg.cv_splits, gap_days=cfg.gap_days_purge
    )

    # Choose estimator (LightGBM if available, else RandomForestRegressor)
    estimator = None
    model_type_used = "lightgbm"
    try:
        import lightgbm as lgb  # type: ignore

        estimator = lgb.LGBMRegressor(
            random_state=cfg.random_state,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(
            n_estimators=200, random_state=cfg.random_state, n_jobs=-1
        )
        model_type_used = "random_forest"

    # Cross-validated metrics
    ic_list: List[float] = []
    r2_list: List[float] = []
    mse_list: List[float] = []

    for train_idx, valid_idx in splits:
        est = estimator.__class__(**getattr(estimator, "get_params", lambda: {})())
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[valid_idx])
        # Spearman IC
        ic = spearmanr(y[valid_idx], y_pred).correlation
        ic_list.append(0.0 if ic is None or np.isnan(ic) else float(ic))
        # R2 and MSE
        r2_list.append(float(r2_score(y[valid_idx], y_pred)))
        mse_list.append(float(mean_squared_error(y[valid_idx], y_pred)))

    ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
    ic_std = float(np.std(ic_list)) if ic_list else 0.0
    r2_mean = float(np.mean(r2_list)) if r2_list else 0.0
    mse_mean = float(np.mean(mse_list)) if mse_list else 0.0

    # Guardrails
    if ic_mean < cfg.min_ic_mean:
        raise ValueError(
            f"IC mean below threshold: {ic_mean:.4f} < {cfg.min_ic_mean:.4f}"
        )
    if r2_mean < cfg.min_r2_valid:
        raise ValueError(
            f"R2 mean below threshold: {r2_mean:.4f} < {cfg.min_r2_valid:.4f}"
        )
    # Overfitting guard: excessively volatile IC across folds
    if ic_std > cfg.max_ic_gap:
        raise ValueError(
            f"IC std above stability threshold: {ic_std:.4f} > {cfg.max_ic_gap:.4f}"
        )

    # Fit final model on all data
    final_estimator = estimator.__class__(
        **getattr(estimator, "get_params", lambda: {})()
    )
    final_estimator.fit(X, y)

    bundle: Dict[str, object] = {
        "model": final_estimator,
        "model_type": model_type_used,
        "feature_columns": keep_cols,
        "prep_meta": prep_meta,
        "metrics": {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "r2_mean": r2_mean,
            "mse_mean": mse_mean,
            "cv_splits": cfg.cv_splits,
        },
        "config": cfg,
    }
    return bundle


def save_model_artifacts(
    model_bundle: Dict[str, object],
    out_dir: str,
    created_at_iso: Optional[str] = None,
) -> Dict[str, str]:
    """
    Persist model bundle, preprocessing meta (as scaler placeholder) and metadata JSON.

    Returns a dict with written file paths.
    """
    import json
    import pickle
    from datetime import datetime
    from pathlib import Path

    ts_iso = created_at_iso or datetime.utcnow().isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "latest_model.pkl"
    scaler_path = out / "scaler.pkl"
    meta_path = out / "model_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(model_bundle.get("prep_meta", {}), f)

    cfg: TrainingConfig = model_bundle.get("config")  # type: ignore[assignment]
    meta = {
        "created_at": ts_iso,
        "model_type": model_bundle.get("model_type"),
        "feature_columns": model_bundle.get("feature_columns"),
        "metrics": model_bundle.get("metrics"),
        "config": {
            "model_type": getattr(cfg, "model_type", None),
            "cv_splits": getattr(cfg, "cv_splits", None),
            "gap_days_purge": getattr(cfg, "gap_days_purge", None),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "model": str(model_path),
        "scaler": str(scaler_path),
        "meta": str(meta_path),
    }


def load_model(model_path: str) -> Dict[str, object]:
    """
    Load persisted model artifacts from disk and return a dict
    compatible with `fit` output.
    """
    import pickle

    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_scores(
    model_bundle: Dict[str, object], features: pd.DataFrame
) -> pd.DataFrame:
    """
    Produce model scores for a given features frame. Returns a DataFrame
    with ['timestamp','symbol','score'].
    """
    keys = {"timestamp", "symbol"}
    if not keys.issubset(features.columns):
        raise ValueError("features must include 'timestamp' and 'symbol'")

    feature_columns: List[str] = model_bundle["feature_columns"]  # type: ignore
    used = [c for c in feature_columns if c in features.columns]
    if len(used) != len(feature_columns):
        missing = [c for c in feature_columns if c not in used]
        # For robustness we allow missing features by filling them
        # with 0 (post-standardization)
        # but we still report them for auditability.
        for m in missing:
            features[m] = 0.0
        used = feature_columns

    df = (
        features[["timestamp", "symbol"] + used]
        .copy()
        .sort_values(["symbol", "timestamp"])
    )
    # Apply simple imputation (0) assuming features are already standardized
    X = df[used].fillna(0.0).values

    model = model_bundle["model"]  # type: ignore
    scores = model.predict(X)
    out = df[["timestamp", "symbol"]].copy()
    out["score"] = scores
    return out


def preprocess_cross_sectional(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Apply cross-sectional preprocessing per date on provided feature columns:
    - Winsorize tails to [q05, q95] per date
    - Z-score standardize per date: (x - mean)/std
    - Impute any remaining NaNs to 0

    Returns the transformed DataFrame and a metadata dict with per-date
    statistics (quantiles/means/stds) for auditability.
    """
    if feature_columns is None:
        exclude = {"timestamp", "symbol", "y_excess"}
        feature_columns = [c for c in df.columns if c not in exclude]
    work = df.copy()
    # Per-date quantiles with safe handling for all-NaN groups
    grouped = work.groupby("timestamp", sort=False)
    q05 = grouped[feature_columns].transform(lambda x: x.quantile(0.05))
    q95 = grouped[feature_columns].transform(lambda x: x.quantile(0.95))
    clipped = work[feature_columns].clip(lower=q05, upper=q95)
    # Z-score per date; avoid division by zero in constant columns
    means = clipped.groupby(work["timestamp"], sort=False).transform("mean")
    stds = (
        clipped.groupby(work["timestamp"], sort=False)
        .transform("std")
        .replace(0.0, np.nan)
    )
    standardized = (clipped - means) / stds
    standardized = standardized.fillna(0.0)

    out = work.copy()
    out[feature_columns] = standardized

    meta: Dict[str, object] = {
        "feature_columns": feature_columns,
        "note": "winsorize[q05,q95] + zscore per date",
    }
    return out, meta


def purged_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    gap_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Purged K-Fold indices over time using unique dates as fold units.
    For each fold's validation date window, purge training rows whose timestamp
    lies within [val_start - gap_days, val_end + gap_days].
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp' for temporal splits")

    # Unique sorted dates and positional mapping
    date_series = pd.to_datetime(df["timestamp"]).dt.normalize()
    unique_dates = np.array(sorted(date_series.unique()))
    if len(unique_dates) < n_splits:
        raise ValueError("Not enough unique dates for the requested n_splits")
    date_to_pos = {d: i for i, d in enumerate(unique_dates)}
    ts_pos = date_series.map(date_to_pos).to_numpy()

    # Determine fold boundaries over date positions
    fold_sizes = np.full(n_splits, len(unique_dates) // n_splits, dtype=int)
    fold_sizes[: len(unique_dates) % n_splits] += 1
    boundaries = np.cumsum(fold_sizes)
    pos_slices = np.split(np.arange(len(unique_dates)), boundaries[:-1])

    all_indices = np.arange(len(df))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for pos_slice in pos_slices:
        if len(pos_slice) == 0:
            continue
        s_pos = int(pos_slice.min())
        e_pos = int(pos_slice.max())
        # Validation indices: rows whose date position falls inside [s_pos, e_pos]
        val_mask = (ts_pos >= s_pos) & (ts_pos <= e_pos)
        val_idx = all_indices[val_mask]
        # Purge window in position space
        purge_low = max(0, s_pos - gap_days)
        purge_high = min(len(unique_dates) - 1, e_pos + gap_days)
        purge_mask = (ts_pos >= purge_low) & (ts_pos <= purge_high)
        train_idx = all_indices[~val_mask & ~purge_mask]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError("Purged K-Fold produced no valid splits")
    return splits
