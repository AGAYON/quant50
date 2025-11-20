import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services import data, execute, features, model, optimize, report

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


class RunResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, object]] = None


class OrderResponse(BaseModel):
    symbol: str
    qty: float
    side: str
    status: str
    filled_at: Optional[datetime]
    filled_avg_price: Optional[float]


def run_pipeline():
    """
    Executes the full quantitative pipeline.
    """
    logger.info("🚀 Starting daily pipeline execution...")
    details = {}

    try:
        # 1. Load Data
        logger.info("Step 1: Loading data from DuckDB...")
        bars = data.get_all_bars()
        if bars.empty:
            logger.warning("⚠️ No data found in DuckDB. Skipping pipeline.")
            return {"error": "No data found"}
        details["bars_count"] = len(bars)

        # 2. Feature Engineering
        logger.info("Step 2: Generating features...")
        feats = features.generate_features(bars)
        if feats.empty:
            logger.warning("⚠️ No features generated.")
            return {"error": "No features generated"}
        details["features_shape"] = feats.shape

        # 3. Model Inference
        logger.info("Step 3: Loading model and predicting...")
        model_path = os.path.join("models", "latest_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Model not found at {model_path}. Skipping inference.")
            return {"error": "Model not found"}

        model_bundle = model.load_model(model_path)
        scores = model.predict_scores(model_bundle, feats)

        # Get latest scores (for today/tomorrow)
        latest_date = scores["timestamp"].max()
        latest_scores = scores[scores["timestamp"] == latest_date].set_index("symbol")[
            "score"
        ]
        details["predictions_count"] = len(latest_scores)

        # 4. Optimization
        logger.info("Step 4: Optimizing portfolio...")
        # Need covariance matrix for optimization
        # We'll use the last N days of returns from bars
        pivot_prices = bars.pivot(index="timestamp", columns="symbol", values="close")
        returns = pivot_prices.pct_change().tail(252)  # Last year
        cov_matrix = optimize.compute_ledoit_wolf_cov(returns)

        # Align covariance with scores
        # Note: optimize_weights handles alignment internally but
        # expects matching indices roughly
        # We'll pass the full covariance and let it select

        # Get previous weights if available (for turnover)
        current_positions = execute.get_current_positions()
        prev_weights = None
        if not current_positions.empty:
            account = execute.get_account()
            equity = float(account.get("equity", 0) or 0)
            if equity > 0:
                current_positions["weight"] = (
                    current_positions["qty"] * current_positions["current_price"]
                ) / equity
                prev_weights = current_positions.set_index("symbol")["weight"]

        opt_result = optimize.optimize_weights(
            expected_returns=latest_scores,
            cov_matrix=cov_matrix,
            prev_weights=prev_weights,
            config=optimize.OptimizeConfig(max_assets=50),
        )
        target_weights = opt_result["weights"]
        details["target_weights_count"] = len(target_weights)

        # 5. Execution
        logger.info("Step 5: Executing orders...")
        exec_result = execute.rebalance_portfolio(target_weights)
        details["execution"] = exec_result

        # 6. Reporting
        logger.info("Step 6: Generating report...")
        rep = report.generate_daily_report()
        pdf_path = report.create_pdf_report(rep)
        details["report_path"] = pdf_path

        logger.info("✅ Pipeline completed successfully.")
        return details

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return {"error": str(e)}


@router.post("/run", response_model=RunResponse)
async def trigger_daily_run(background_tasks: BackgroundTasks):
    """
    Triggers the complete daily pipeline in the background.
    """
    background_tasks.add_task(run_pipeline)
    return RunResponse(
        status="accepted",
        message="Daily pipeline triggered in background",
        timestamp=datetime.utcnow(),
    )


@router.post("/report", response_model=RunResponse)
def generate_report():
    """
    Manually triggers the generation of the daily PDF report.
    """
    try:
        rep = report.generate_daily_report()
        pdf_path = report.create_pdf_report(rep)
        return RunResponse(
            status="success",
            message=f"Report generated at {pdf_path}",
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders", response_model=List[OrderResponse])
def get_orders(limit: int = 50):
    """
    Returns a list of recent orders from Alpaca.
    """
    try:
        orders_df = execute.get_recent_orders(limit=limit)
        if orders_df.empty:
            return []

        # Convert DataFrame to list of dicts
        orders_list = orders_df.to_dict("records")

        # Map to response model
        response = []
        for o in orders_list:
            response.append(
                OrderResponse(
                    symbol=o.get("symbol"),
                    qty=float(o.get("qty", 0)),
                    side=o.get("side"),
                    status=o.get("status"),
                    filled_at=o.get("filled_at"),
                    filled_avg_price=(
                        float(o.get("filled_avg_price", 0))
                        if o.get("filled_avg_price")
                        else None
                    ),
                )
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
