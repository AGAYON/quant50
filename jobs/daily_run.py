"""
Daily run script for Quant50.

This script executes the full quantitative pipeline:
1. Data Ingestion
2. Feature Engineering
3. Model Inference
4. Portfolio Optimization
5. Execution
6. Reporting

It is designed to be run via cron or GitHub Actions.
"""

import logging
import sys
from datetime import datetime

from app.services import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("🎬 Starting daily run job...")
    start_time = datetime.utcnow()

    result = pipeline.run_pipeline()

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    if "error" in result:
        logger.error(f"❌ Daily run failed after {duration:.2f}s: {result['error']}")
        return 1

    logger.info(f"✅ Daily run completed successfully in {duration:.2f}s")
    logger.info(f"Details: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
