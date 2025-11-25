"""
Weekly training job for Quant50 (T004G).

CLI entrypoint to run the training pipeline and persist artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from app.services.features import generate_features
from app.services.label import build_labels_excess_return
from app.services.model import (
    TrainingConfig,
    build_training_dataset,
    fit,
    save_model_artifacts,
)
from app.utils.config import DUCKDB_PATH, REPORTS_DIR


def _load_bars_from_duckdb() -> pd.DataFrame:
    if not os.path.exists(DUCKDB_PATH):
        return pd.DataFrame()
    con = duckdb.connect(DUCKDB_PATH)
    try:
        exists = con.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_name = 'bars_daily'
            """
        ).fetchone()[0]
        if exists == 0:
            return pd.DataFrame()
        df = con.execute("SELECT * FROM bars_daily").fetchdf()
        return df
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant50 weekly training job")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without saving artifacts"
    )
    parser.add_argument(
        "--benchmark", default="SPY", help="Benchmark symbol for excess returns"
    )
    parser.add_argument(
        "--allow-weak",
        action="store_true",
        help="Relax training thresholds (useful for synthetic/smoke runs)",
    )
    args = parser.parse_args()

    ts = datetime.utcnow()
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load bars
    bars = _load_bars_from_duckdb()
    if bars.empty:
        result = {
            "status": "no_data",
            "timestamp": ts.isoformat(),
            "message": "No bars found in DuckDB; skipping training",
        }
        print(json.dumps(result))
        return 0

    # Build features and labels
    feats = generate_features(bars).reset_index()
    labs = build_labels_excess_return(
        bars, benchmark_symbol=args.benchmark, horizon_days=5
    )
    train_df = build_training_dataset(feats, labs)

    # Fit model
    # Default strict; optionally relax for synthetic/smoke runs
    if args.allow_weak:
        cfg = TrainingConfig(min_ic_mean=-1.0, min_r2_valid=-1.0, max_ic_gap=999.0)
    else:
        cfg = TrainingConfig()
    bundle = fit(train_df, config=cfg)

    # Persist artifacts
    if not args.dry_run:
        save_model_artifacts(bundle, str(models_dir), created_at_iso=ts.isoformat())

        # Training report
        report_path = Path(REPORTS_DIR) / f"train_report_{ts:%Y%m%d}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": ts.isoformat(),
                    "metrics": bundle.get("metrics"),
                    "rows": int(len(train_df)),
                    "symbols": int(train_df["symbol"].nunique()),
                },
                f,
                indent=2,
            )

        # Save Market Snapshot for Release (T011)
        # We need ~252 trading days for features. Saving 365 calendar days is safe.
        snapshot_path = Path("data/market_snapshot.parquet")
        if not bars.empty:
            max_date = bars["timestamp"].max()
            cutoff_date = max_date - pd.Timedelta(days=365)
            snapshot_df = bars[bars["timestamp"] >= cutoff_date].copy()

            # Ensure data directory exists
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_df.to_parquet(snapshot_path)
            print(f"Snapshot saved to {snapshot_path} ({len(snapshot_df)} rows)")

    result = {
        "status": "ok",
        "timestamp": ts.isoformat(),
        "message": "weekly training completed",
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
