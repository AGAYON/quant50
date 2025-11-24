import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services import execute, pipeline, report

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


@router.post("/run", response_model=RunResponse)
async def trigger_daily_run(background_tasks: BackgroundTasks):
    """
    Triggers the complete daily pipeline in the background.
    """
    background_tasks.add_task(pipeline.run_pipeline)
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
