"""
Data Loader Service for Quant50 (T012).

Handles the "Just-in-Time" data strategy:
1. Downloads the latest weekly snapshot (market data + model) from GitHub Releases.
2. Fetches the current day's bar from Alpaca.
3. Merges them in-memory to provide the full history needed for feature engineering.
"""

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.services.data import fetch_stock_data_alpaca

logger = logging.getLogger(__name__)


def download_latest_snapshot(
    output_dir: str = "data", model_dir: str = "models"
) -> bool:
    """
    Download artifacts from the latest GitHub Release using 'gh' CLI.

    Artifacts expected:
    - market_snapshot.parquet -> data/
    - latest_model.pkl -> models/
    - scaler.pkl -> models/
    - model_meta.json -> models/
    """
    logger.info("⬇️ Downloading latest snapshot from GitHub Releases...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Check if gh is available
        subprocess.run(["gh", "--version"], check=True, capture_output=True)

        # Download assets
        # We download everything to a temp dir or directly to locations?
        # gh release download --pattern "*" --dir ...
        # Since we have different targets, let's download to a temp folder first.
        temp_dir = Path("temp_assets")
        temp_dir.mkdir(exist_ok=True)

        cmd = ["gh", "release", "download", "--pattern", "*", "--dir", str(temp_dir)]
        subprocess.run(cmd, check=True, capture_output=True)

        # Move files
        # Data
        if (temp_dir / "market_snapshot.parquet").exists():
            (temp_dir / "market_snapshot.parquet").rename(
                Path(output_dir) / "market_snapshot.parquet"
            )

        # Models
        for f in ["latest_model.pkl", "scaler.pkl", "model_meta.json"]:
            if (temp_dir / f).exists():
                (temp_dir / f).rename(Path(model_dir) / f)

        # Cleanup
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()

        logger.info("✅ Snapshot assets downloaded successfully.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download release assets: {e}")
        return False
    except FileNotFoundError:
        logger.warning(
            "⚠️ 'gh' CLI not found. "
            "Skipping download (assuming local dev or manual setup)."
        )
        return False
    except Exception as e:
        logger.error(f"❌ Error processing snapshot assets: {e}")
        return False


def load_and_merge_data(
    symbols: list[str], snapshot_path: str = "data/market_snapshot.parquet"
) -> pd.DataFrame:
    """
    Load historical snapshot and merge with today's live data.

    Returns
    -------
    pd.DataFrame
        Combined history (snapshot + today) for feature engineering.
    """
    # 1. Load Snapshot
    if not os.path.exists(snapshot_path):
        logger.warning(f"⚠️ Snapshot not found at {snapshot_path}. Returning empty.")
        return pd.DataFrame()

    logger.info(f"📖 Loading snapshot from {snapshot_path}...")
    history = pd.read_parquet(snapshot_path)

    if history.empty:
        return pd.DataFrame()

    # 2. Fetch Today's Data (Incremental)
    # We need to know what 'today' means. Usually run before market close or after?
    # Strategy: fetch from last snapshot date + 1 day until now.

    last_snapshot_date = history["timestamp"].max()
    start_date = (last_snapshot_date + timedelta(days=1)).date()
    today = datetime.utcnow().date()

    if start_date > today:
        logger.info("✅ Snapshot is up to date. No incremental fetch needed.")
        return history

    logger.info(f"🔄 Fetching incremental data from {start_date} to {today}...")

    # We need to fetch for all symbols present in the snapshot (or a defined universe)
    # For now, we use symbols found in history
    universe = history["symbol"].unique()

    new_bars = []
    for symbol in universe:
        # We can use the existing fetch function
        # Note: fetch_stock_data_alpaca prints errors,
        # we might want to suppress or handle them
        df = fetch_stock_data_alpaca(symbol, str(start_date), str(today))
        if not df.empty:
            new_bars.append(df)

    if not new_bars:
        logger.info(
            "⚠️ No new data found for today "
            "(market closed or API delay). Using snapshot only."
        )
        return history

    incremental_df = pd.concat(new_bars, ignore_index=True)
    logger.info(f"✅ Fetched {len(incremental_df)} new bars.")

    # 3. Merge
    combined = pd.concat([history, incremental_df], ignore_index=True)

    # Deduplicate just in case (keep latest)
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return combined
