from datetime import datetime
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Quant50 API running 🚀"


@patch("app.routes.execute.get_recent_orders")
def test_get_orders(mock_get_orders):
    # Mock data
    mock_df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "status": "filled",
                "filled_at": datetime.now(),
                "filled_avg_price": 150.0,
            }
        ]
    )
    mock_get_orders.return_value = mock_df

    response = client.get("/orders")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "AAPL"
    assert data[0]["qty"] == 10.0


@patch("app.routes.report.generate_daily_report")
@patch("app.routes.report.create_pdf_report")
def test_generate_report(mock_create_pdf, mock_generate_report):
    mock_generate_report.return_value = {}
    mock_create_pdf.return_value = "/tmp/report.pdf"

    response = client.post("/report")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "Report generated" in response.json()["message"]


@patch("app.services.pipeline.run_pipeline")
def test_trigger_run(mock_run_pipeline):
    # We mock the background task function itself to verify it's called
    # But since it's a background task, we just check the response

    response = client.post("/run")
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
