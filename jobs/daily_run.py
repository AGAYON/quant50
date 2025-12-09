"""
Daily run script for Quant50.

This script executes the full quantitative pipeline with Fail-Safe logic (T012/T013):
1. Downloads weekly snapshot (market data + model) from GitHub Releases.
2. Fetches today's incremental data from Alpaca.
3. Merges data in-memory (no local DB persistence required).
4. Executes pipeline (Inference -> Optimize -> Execute -> Report).
5. Handles errors by cancelling open orders and generating an error report.
"""

import logging
import sys
import traceback
from datetime import datetime

from app.services import pipeline
from app.services.data_loader import download_latest_snapshot, load_and_merge_data
from app.services.execute import cancel_all_orders
from app.services.report import generate_error_report
from app.services.validation import check_market_open, validate_model_age

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("🎬 Starting daily run job (Snapshot + Incremental)...")
    start_time = datetime.utcnow()

    try:
        # --- Step 0: Validation (T015) ---
        # 1. Market Open Check
        is_open, msg = check_market_open()
        if not is_open:
            logger.info(f"⏸️ Market is closed: {msg}. Skipping run gracefully.")
            return 0
        logger.info(f"✅ Market check passed: {msg}")

        # --- Step 1: Data Ingestion (T012) ---
        # Download snapshot from Release
        if not download_latest_snapshot():
            raise RuntimeError("Failed to download weekly snapshot.")

        # 2. Model Age Check (after download)
        # We expect models/latest_model.pkl to be present now
        is_fresh, msg = validate_model_age("models/latest_model.pkl", max_days=6)
        if not is_fresh:
            raise RuntimeError(f"Model validation failed: {msg}")
        logger.info(f"✅ Model check passed: {msg}")

        # Load snapshot and merge with today's live data
        # We need a universe of symbols. For now, we rely on what's in the snapshot.
        # If snapshot is empty, we can't proceed.
        bars = load_and_merge_data(symbols=[])
        if bars.empty:
            raise RuntimeError(
                "No market data available (snapshot empty + no live data)."
            )

        # --- Step 2: Pipeline Execution ---
        # Inject the merged bars into the pipeline
        result = pipeline.run_pipeline(bars=bars)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        if "error" in result:
            raise RuntimeError(f"Pipeline reported error: {result['error']}")

        logger.info(f"✅ Daily run completed successfully in {duration:.2f}s")
        logger.info(f"Details: {result}")
        return 0

    except Exception as e:
        # --- Step 3: Fail-Safe Handling (T013) ---
        logger.error(f"❌ CRITICAL FAILURE: {e}", exc_info=True)

        # 1. Safety Halt: Cancel all open orders
        logger.warning("🛑 Initiating Safety Halt...")
        cancelled = cancel_all_orders()
        logger.info(f"Safety Halt complete: {cancelled} orders cancelled.")

        # 2. Generate Error Report
        error_msg = f"Exception: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        report_path = generate_error_report(error_msg)
        logger.info(f"📄 Error report generated at: {report_path}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
