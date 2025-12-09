"""
Unit tests for daily report generation (T007).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.report import (
    calculate_diversification_index,
    calculate_order_metrics,
    calculate_pnl_metrics,
    calculate_risk_metrics,
    create_pdf_report,
    generate_daily_report,
    generate_error_report,
    get_account_history,
    get_recent_orders,
    get_top_holdings,
)


@pytest.fixture
def mock_account_response():
    """Mock Alpaca account response."""
    return {
        "account_number": "ABC123",
        "equity": "100000.00",
        "cash": "50000.00",
        "buying_power": "100000.00",
    }


@pytest.fixture
def mock_positions_response():
    """Mock Alpaca positions response."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "qty": [10.0, 5.0, 3.0],
            "market_value": [1500.0, 1500.0, 4500.0],
            "avg_entry_price": [150.0, 300.0, 1500.0],
            "current_price": [150.0, 300.0, 1500.0],
        }
    )


@pytest.fixture
def mock_equity_history():
    """Mock equity history DataFrame."""
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    equity_values = [100000.0 + i * 100.0 for i in range(30)]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity_values,
            "cash": [50000.0] * 30,
            "buying_power": [100000.0] * 30,
        }
    )


@patch("app.services.report.requests.get")
def test_get_account_history_success(mock_get: MagicMock):
    """Test successful account history retrieval."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "timestamp": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
        "equity": [100000.0, 101000.0],
        "cash": [50000.0, 50000.0],
        "buying_power": [100000.0, 100000.0],
    }
    mock_get.return_value = mock_response

    result = get_account_history()
    assert len(result) > 0
    assert "equity" in result.columns
    assert "timestamp" in result.columns


@patch("app.services.report.requests.get")
def test_get_account_history_error(mock_get: MagicMock):
    """Test account history with API error."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    result = get_account_history()
    assert result.empty or "equity" in result.columns


@patch("app.services.report.requests.get")
def test_get_recent_orders_success(mock_get: MagicMock):
    """Test successful orders retrieval."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "status": "filled",
            "filled_at": "2025-01-01T12:00:00Z",
            "filled_avg_price": "150.00",
        }
    ]
    mock_get.return_value = mock_response

    result = get_recent_orders(limit=10)
    assert len(result) > 0
    assert "symbol" in result.columns


def test_calculate_pnl_metrics(mock_equity_history: pd.DataFrame):
    """Test PnL metrics calculation."""
    metrics = calculate_pnl_metrics(mock_equity_history)
    assert "daily_pnl" in metrics
    assert "cumulative_pnl" in metrics
    assert "ytd_return" in metrics
    assert "cagr" in metrics
    assert "current_equity" in metrics
    assert "starting_equity" in metrics
    assert metrics["current_equity"] > 0


def test_calculate_pnl_metrics_empty():
    """Test PnL metrics with empty history."""
    metrics = calculate_pnl_metrics(pd.DataFrame())
    assert metrics["daily_pnl"] == 0.0
    assert metrics["cumulative_pnl"] == 0.0


def test_calculate_risk_metrics(mock_equity_history: pd.DataFrame):
    """Test risk metrics calculation."""
    metrics = calculate_risk_metrics(mock_equity_history)
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "turnover" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "var_95" in metrics
    assert "cvar_95" in metrics
    assert "rolling_volatility_30d" in metrics
    assert isinstance(metrics["max_drawdown"], float)
    assert isinstance(metrics["volatility"], float)


def test_calculate_risk_metrics_empty():
    """Test risk metrics with empty history."""
    metrics = calculate_risk_metrics(pd.DataFrame())
    assert metrics["max_drawdown"] == 0.0
    assert metrics["volatility"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["var_95"] == 0.0
    assert metrics["cvar_95"] == 0.0


def test_get_top_holdings(mock_positions_response: pd.DataFrame):
    """Test top holdings extraction."""
    result = get_top_holdings(mock_positions_response, n=5)
    assert "top_holdings" in result
    assert "sector_weights" in result
    assert len(result["top_holdings"]) <= 5
    assert "symbol" in result["top_holdings"].columns
    assert "weight" in result["top_holdings"].columns


def test_get_top_holdings_with_sectors(mock_positions_response: pd.DataFrame):
    """Test top holdings with sector mapping."""
    symbol_to_sector = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech"}
    result = get_top_holdings(
        mock_positions_response, n=5, symbol_to_sector=symbol_to_sector
    )
    assert "sector_weights" in result
    assert len(result["sector_weights"]) > 0
    assert "sector" in result["sector_weights"].columns


def test_get_top_holdings_empty():
    """Test top holdings with empty positions."""
    result = get_top_holdings(pd.DataFrame(), n=5)
    assert result["top_holdings"].empty
    assert result["sector_weights"].empty


@patch("app.services.report.get_account")
@patch("app.services.report.get_current_positions")
@patch("app.services.report.get_account_history_with_fallback")
@patch("app.services.report.get_recent_orders")
def test_generate_daily_report_success(
    mock_get_orders: MagicMock,
    mock_get_history: MagicMock,
    mock_get_positions: MagicMock,
    mock_get_account: MagicMock,
    mock_account_response: dict,
    mock_positions_response: pd.DataFrame,
    mock_equity_history: pd.DataFrame,
):
    """Test successful daily report generation."""
    mock_get_account.return_value = mock_account_response
    mock_get_positions.return_value = mock_positions_response
    mock_get_history.return_value = (mock_equity_history, "api_success")
    mock_get_orders.return_value = pd.DataFrame()

    report = generate_daily_report()

    assert "report_date" in report
    assert "account" in report
    assert "pnl_metrics" in report
    assert "risk_metrics" in report
    assert "top_holdings" in report
    assert "sector_weights" in report
    assert "recent_orders" in report
    assert "diversification_index" in report
    assert "order_metrics" in report
    assert "data_source_status" in report
    assert report["data_source_status"] == "api_success"


def test_create_pdf_report(tmp_path, mock_account_response):
    """Test PDF report creation."""
    report_data = {
        "report_date": "2025-01-01",
        "account": mock_account_response,
        "pnl_metrics": {
            "daily_pnl": 100.0,
            "cumulative_pnl": 1000.0,
            "ytd_return": 5.0,
            "current_equity": 105000.0,
            "starting_equity": 100000.0,
        },
        "risk_metrics": {
            "max_drawdown": -2.0,
            "volatility": 15.0,
            "turnover": 10.0,
        },
        "top_holdings": [
            {"symbol": "AAPL", "market_value": 1500.0, "weight": 1.5},
            {"symbol": "MSFT", "market_value": 1500.0, "weight": 1.5},
        ],
        "sector_weights": [{"sector": "Tech", "weight": 50.0}],
        "positions_count": 2,
        "recent_orders": [
            {"symbol": "AAPL", "side": "buy", "qty": "10", "status": "filled"}
        ],
        "orders_count": 1,
        "data_source_status": "api_success",
    }

    output_path = tmp_path / "test_report.pdf"
    result_path = create_pdf_report(report_data, str(output_path), include_charts=False)

    assert result_path == str(output_path)
    assert output_path.exists()


def test_create_pdf_report_minimal(tmp_path):
    """Test PDF creation with minimal data."""
    report_data = {
        "report_date": "2025-01-01",
        "account": {"equity": 0.0, "cash": 0.0, "buying_power": 0.0},
        "pnl_metrics": {
            "daily_pnl": 0.0,
            "cumulative_pnl": 0.0,
            "ytd_return": 0.0,
            "current_equity": 0.0,
            "starting_equity": 0.0,
        },
        "risk_metrics": {"max_drawdown": 0.0, "volatility": 0.0, "turnover": 0.0},
        "top_holdings": [],
        "sector_weights": [],
        "positions_count": 0,
        "recent_orders": [],
        "orders_count": 0,
        "data_source_status": "api_success",
    }

    output_path = tmp_path / "test_minimal.pdf"
    _ = create_pdf_report(report_data, str(output_path), include_charts=False)
    assert output_path.exists()


def test_calculate_diversification_index(mock_positions_response: pd.DataFrame):
    """Test HHI diversification index calculation."""
    hhi = calculate_diversification_index(mock_positions_response)
    assert 0.0 <= hhi <= 1.0
    assert isinstance(hhi, float)


def test_calculate_diversification_index_empty():
    """Test HHI with empty positions."""
    hhi = calculate_diversification_index(pd.DataFrame())
    assert hhi == 1.0  # Maximum concentration


def test_calculate_order_metrics():
    """Test order metrics calculation."""
    orders = pd.DataFrame(
        {
            "status": ["filled", "filled", "rejected", "pending"],
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        }
    )
    metrics = calculate_order_metrics(orders)
    assert "fill_rate" in metrics
    assert "avg_holding_period_days" in metrics
    assert metrics["fill_rate"] == 0.5  # 2 out of 4 filled


def test_calculate_order_metrics_empty():
    """Test order metrics with empty orders."""
    metrics = calculate_order_metrics(pd.DataFrame())
    assert metrics["fill_rate"] == 0.0
    assert metrics["avg_holding_period_days"] == 0.0


@patch("app.services.report.get_account")
def test_generate_error_report(mock_get_account, tmp_path):
    """Test error report generation."""
    mock_get_account.return_value = {"equity": "100000.00", "cash": "50000.00"}

    output_path = tmp_path / "error_report.pdf"
    result_path = generate_error_report("Something went wrong", str(output_path))

    assert result_path == str(output_path)
    assert output_path.exists()
