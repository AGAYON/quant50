"""
Unit tests for chart generation (T007B).
"""

import os

import pandas as pd
import pytest

from app.services.report_charts import (
    generate_all_charts,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_sector_weights,
    plot_top_holdings,
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
        }
    )


@pytest.fixture
def mock_sector_weights():
    """Mock sector weights DataFrame."""
    return pd.DataFrame(
        {
            "sector": ["Tech", "Finance", "Healthcare"],
            "weight": [40.0, 30.0, 30.0],
        }
    )


@pytest.fixture
def mock_top_holdings():
    """Mock top holdings DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "weight": [15.0, 12.0, 10.0, 8.0, 5.0],
        }
    )


def test_plot_equity_curve(mock_equity_history: pd.DataFrame, tmp_path):
    """Test equity curve chart generation."""
    output_path = tmp_path / "equity_curve.png"
    result_path = plot_equity_curve(mock_equity_history, str(output_path))

    assert result_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_equity_curve_empty(tmp_path):
    """Test equity curve with empty data."""
    output_path = tmp_path / "equity_empty.png"
    plot_equity_curve(pd.DataFrame(), str(output_path))

    assert output_path.exists()


def test_plot_drawdown_curve(mock_equity_history: pd.DataFrame, tmp_path):
    """Test drawdown curve chart generation."""
    output_path = tmp_path / "drawdown.png"
    result_path = plot_drawdown_curve(mock_equity_history, str(output_path))

    assert result_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_drawdown_curve_empty(tmp_path):
    """Test drawdown curve with empty data."""
    output_path = tmp_path / "drawdown_empty.png"
    plot_drawdown_curve(pd.DataFrame(), str(output_path))

    assert output_path.exists()


def test_plot_sector_weights(mock_sector_weights: pd.DataFrame, tmp_path):
    """Test sector weights chart generation."""
    output_path = tmp_path / "sector_weights.png"
    result_path = plot_sector_weights(mock_sector_weights, str(output_path))

    assert result_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_sector_weights_empty(tmp_path):
    """Test sector weights with empty data."""
    output_path = tmp_path / "sector_empty.png"
    plot_sector_weights(pd.DataFrame(), str(output_path))

    assert output_path.exists()


def test_plot_top_holdings(mock_top_holdings: pd.DataFrame, tmp_path):
    """Test top holdings chart generation."""
    output_path = tmp_path / "top_holdings.png"
    result_path = plot_top_holdings(mock_top_holdings, str(output_path), n=5)

    assert result_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_top_holdings_empty(tmp_path):
    """Test top holdings with empty data."""
    output_path = tmp_path / "holdings_empty.png"
    plot_top_holdings(pd.DataFrame(), str(output_path), n=10)

    assert output_path.exists()


def test_generate_all_charts(
    mock_equity_history: pd.DataFrame,
    mock_sector_weights: pd.DataFrame,
    mock_top_holdings: pd.DataFrame,
    tmp_path,
):
    """Test generation of all charts."""
    chart_dir = tmp_path / "charts"
    chart_paths = generate_all_charts(
        mock_equity_history,
        mock_sector_weights,
        mock_top_holdings,
        str(chart_dir),
    )

    assert "equity_curve" in chart_paths
    assert "drawdown" in chart_paths
    assert "sector_weights" in chart_paths
    assert "top_holdings" in chart_paths

    # Verify all charts were created
    for chart_path in chart_paths.values():
        assert os.path.exists(chart_path)
        assert os.path.getsize(chart_path) > 0


def test_generate_all_charts_empty_data(tmp_path):
    """Test chart generation with empty data."""
    chart_dir = tmp_path / "charts"
    chart_paths = generate_all_charts(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        str(chart_dir),
    )

    # Should still generate equity and drawdown charts (with placeholder)
    assert "equity_curve" in chart_paths
    assert "drawdown" in chart_paths
    # Sector and holdings may be skipped if empty
    assert os.path.exists(chart_paths["equity_curve"])
