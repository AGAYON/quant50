import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services import pipeline


@pytest.fixture
def mock_env():
    # Create a temporary directory for reports and db
    temp_dir = tempfile.mkdtemp()
    reports_dir = os.path.join(temp_dir, "reports")
    os.makedirs(reports_dir)
    db_path = os.path.join(temp_dir, "test_market.duckdb")

    # Patch config variables
    with (
        patch("app.services.data.DUCKDB_PATH", db_path),
        patch("app.services.report.REPORTS_DIR", reports_dir),
        patch("app.utils.config.DUCKDB_PATH", db_path),
        patch("app.utils.config.REPORTS_DIR", reports_dir),
    ):
        yield {"db_path": db_path, "reports_dir": reports_dir}

    # Cleanup
    shutil.rmtree(temp_dir)


@patch("app.services.data.get_latest_bars")
@patch("app.services.data.get_all_bars")
@patch("app.services.model.load_model")
@patch("app.services.model.predict_scores")
@patch("app.services.report.get_current_positions")
@patch("app.services.execute.get_current_positions")
@patch("app.services.execute.get_account")
@patch("app.services.report.get_account")
@patch("app.services.execute.rebalance_portfolio")
def test_full_pipeline_e2e(
    mock_rebalance,
    mock_report_get_account,
    mock_get_account,
    mock_get_positions,
    mock_predict,
    mock_load_model,
    mock_get_bars,
    mock_get_latest_bars,
    mock_env,
):
    # 1. Setup Mock Data
    # Mock Bars
    dates = pd.date_range(end=pd.Timestamp.now(), periods=300, freq="B")
    data = {
        "timestamp": dates,
        "symbol": "AAPL",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000000,
    }
    bars_df = pd.DataFrame(data)
    # Add a second symbol to ensure covariance calculation works
    data2 = data.copy()
    data2["symbol"] = "GOOGL"
    bars_df = pd.concat([bars_df, pd.DataFrame(data2)])

    mock_get_latest_bars.return_value = bars_df
    mock_get_bars.return_value = bars_df

    # Mock Model
    mock_load_model.return_value = {"model": MagicMock()}

    # Mock Predictions
    scores_df = pd.DataFrame(
        {
            "timestamp": [dates[-1], dates[-1]],
            "symbol": ["AAPL", "GOOGL"],
            "score": [0.05, 0.03],
        }
    )
    mock_predict.return_value = scores_df

    # Mock Execution
    mock_get_positions.return_value = pd.DataFrame()  # No current positions
    mock_get_positions.return_value = pd.DataFrame()  # No current positions
    mock_get_account.return_value = {"equity": "100000", "cash": "100000"}
    mock_report_get_account.return_value = {"equity": "100000", "cash": "100000"}
    mock_rebalance.return_value = {"orders": []}

    # 2. Run Pipeline
    result = pipeline.run_pipeline()

    # 3. Verification
    assert "error" not in result, f"Pipeline failed with error: {result.get('error')}"
    assert result.get("bars_count") == 600  # 300 * 2
    assert "features_shape" in result
    assert result.get("predictions_count") == 2
    assert result.get("target_weights_count") > 0
    assert "report_path" in result

    # Verify report file exists
    report_path = result["report_path"]
    assert os.path.exists(report_path)
    assert report_path.endswith(".pdf")
