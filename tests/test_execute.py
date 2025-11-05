"""
Unit tests for order execution services (T006).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.execute import (
    ExecutionConfig,
    compute_order_deltas,
    get_account,
    get_current_positions,
    rebalance_portfolio,
    send_order,
    send_orders,
    validate_execution,
)


@pytest.fixture
def mock_account_response():
    """Mock Alpaca account response."""
    return {
        "account_number": "ABC123",
        "buying_power": "100000.00",
        "cash": "50000.00",
        "equity": "100000.00",
        "portfolio_value": "100000.00",
    }


@pytest.fixture
def mock_positions_response():
    """Mock Alpaca positions response."""
    return [
        {
            "symbol": "AAPL",
            "qty": "10",
            "market_value": "1500.00",
            "avg_entry_price": "150.00",
            "current_price": "150.00",
        },
        {
            "symbol": "MSFT",
            "qty": "5",
            "market_value": "1500.00",
            "avg_entry_price": "300.00",
            "current_price": "300.00",
        },
    ]


@pytest.fixture
def mock_order_response():
    """Mock Alpaca order response."""
    return {
        "id": "order-123",
        "symbol": "AAPL",
        "qty": "5",
        "side": "buy",
        "type": "market",
        "status": "new",
    }


@patch("app.services.execute.requests.get")
def test_get_account_success(mock_get: MagicMock, mock_account_response):
    """Test successful account retrieval."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_account_response
    mock_get.return_value = mock_response

    result = get_account()
    assert result["equity"] == "100000.00"
    assert result["buying_power"] == "100000.00"
    mock_get.assert_called_once()


@patch("app.services.execute.requests.get")
def test_get_account_unauthorized(mock_get: MagicMock):
    """Test account retrieval with 401 error."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_get.return_value = mock_response

    with pytest.raises(ValueError, match="Unauthorized"):
        get_account()


@patch("app.services.execute.requests.get")
def test_get_current_positions_success(mock_get: MagicMock, mock_positions_response):
    """Test successful positions retrieval."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_positions_response
    mock_get.return_value = mock_response

    result = get_current_positions()
    assert len(result) == 2
    assert "AAPL" in result["symbol"].values
    assert "MSFT" in result["symbol"].values
    assert result["qty"].dtype == float
    mock_get.assert_called_once()


@patch("app.services.execute.requests.get")
def test_get_current_positions_empty(mock_get: MagicMock):
    """Test positions retrieval with no positions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    result = get_current_positions()
    assert len(result) == 0
    assert list(result.columns) == [
        "symbol",
        "qty",
        "market_value",
        "avg_entry_price",
        "current_price",
    ]


def test_compute_order_deltas_buy():
    """Test computing order deltas for new positions (buy)."""
    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "weight": [0.5, 0.5],
        }
    )

    # Create positions with prices (AAPL exists, MSFT doesn't)
    current_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [10.0],
            "current_price": [150.0],
        }
    )

    account_value = 100000.0

    deltas = compute_order_deltas(target_weights, current_positions, account_value)

    # AAPL should have a target of 50k (333.33 shares),
    # current is 1.5k (10 shares) → BUY
    # MSFT has no current_price, so target_qty will be 0
    # (handled by price_mask logic)
    assert len(deltas) > 0
    # AAPL should definitely be in deltas (has position and price)
    aapl_delta = deltas[deltas["symbol"] == "AAPL"]
    assert len(aapl_delta) > 0
    assert aapl_delta.iloc[0]["action"] == "BUY"
    assert aapl_delta.iloc[0]["delta_qty"] > 0


def test_compute_order_deltas_sell():
    """Test computing order deltas for position reduction (sell)."""
    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "weight": [0.1],  # Reduce from current
        }
    )

    current_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [100.0],  # Large position
            "current_price": [150.0],
        }
    )

    account_value = 100000.0

    deltas = compute_order_deltas(target_weights, current_positions, account_value)

    # Should have a SELL for AAPL
    assert len(deltas) > 0
    aapl_delta = deltas[deltas["symbol"] == "AAPL"].iloc[0]
    assert aapl_delta["action"] == "SELL"
    assert aapl_delta["delta_qty"] < 0


def test_compute_order_deltas_no_rebalance():
    """Test compute_order_deltas when no rebalancing is needed."""
    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "weight": [0.5],
        }
    )

    # Current position matches target (approximately)
    current_positions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [333.33],  # ~50k at 150 per share
            "current_price": [150.0],
        }
    )

    account_value = 100000.0

    deltas = compute_order_deltas(target_weights, current_positions, account_value)

    # Should be empty or HOLD (depending on tolerance)
    # The function filters out HOLD actions, so result should be empty or small
    assert len(deltas) == 0 or all(
        abs(d["delta_qty"]) < 1e-3 for d in deltas.to_dict("records")
    )


@patch("app.services.execute.requests.post")
def test_send_order_success(mock_post: MagicMock, mock_order_response):
    """Test successful order sending."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_order_response
    mock_post.return_value = mock_response

    result = send_order("AAPL", 5.0, "buy")
    assert result["id"] == "order-123"
    assert result["symbol"] == "AAPL"
    mock_post.assert_called_once()


@patch("app.services.execute.requests.post")
def test_send_order_dry_run(mock_post: MagicMock):
    """Test order sending in dry-run mode."""
    config = ExecutionConfig(dry_run=True)
    result = send_order("AAPL", 5.0, "buy", config=config)
    assert result["status"] == "dry_run"
    assert result["symbol"] == "AAPL"
    mock_post.assert_not_called()


def test_send_order_invalid_side():
    """Test send_order with invalid side."""
    with pytest.raises(ValueError, match="side must be"):
        send_order("AAPL", 5.0, "invalid")


@patch("app.services.execute.send_order")
def test_send_orders_multiple(mock_send_order: MagicMock):
    """Test sending multiple orders."""
    mock_send_order.return_value = {"status": "filled", "id": "order-1"}

    deltas = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "action": ["BUY", "SELL"],
            "delta_qty": [10.0, -5.0],
            "delta_value_usd": [1500.0, -1500.0],
        }
    )

    results = send_orders(deltas)
    assert len(results) == 2
    assert mock_send_order.call_count == 2


@patch("app.services.execute.get_account")
@patch("app.services.execute.get_current_positions")
def test_validate_execution_pass(
    mock_get_positions: MagicMock,
    mock_get_account: MagicMock,
    mock_account_response,
    mock_positions_response,
):
    """Test execution validation when positions match targets."""
    mock_get_account.return_value = mock_account_response
    mock_get_positions.return_value = pd.DataFrame(mock_positions_response)

    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "weight": [0.015, 0.015],  # ~1.5k each out of 100k
        }
    )

    result = validate_execution(target_weights, tolerance=0.05)
    assert result["valid"] is True
    assert result["max_deviation"] <= 0.05


@patch("app.services.execute.get_account")
@patch("app.services.execute.get_current_positions")
def test_validate_execution_fail(
    mock_get_positions: MagicMock,
    mock_get_account: MagicMock,
    mock_account_response,
):
    """Test execution validation when positions don't match targets."""
    mock_get_account.return_value = mock_account_response
    # Positions with wrong weights
    mock_get_positions.return_value = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "qty": 1000.0,  # Way too much
                "current_price": 150.0,
            }
        ]
    )

    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "weight": [0.01],  # Should be ~1k, not 150k
        }
    )

    result = validate_execution(target_weights, tolerance=0.05)
    # Should fail due to large deviation
    assert result["valid"] is False or result["max_deviation"] > 0.05


@patch("app.services.execute.send_orders")
@patch("app.services.execute.compute_order_deltas")
@patch("app.services.execute.get_current_positions")
@patch("app.services.execute.get_account")
def test_rebalance_portfolio_success(
    mock_get_account: MagicMock,
    mock_get_positions: MagicMock,
    mock_compute_deltas: MagicMock,
    mock_send_orders: MagicMock,
    mock_account_response,
):
    """Test complete rebalancing workflow."""
    mock_get_account.return_value = mock_account_response
    mock_get_positions.return_value = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "qty": [10.0],
            "current_price": [150.0],
        }
    )

    deltas_df = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "action": ["BUY"],
            "delta_qty": [10.0],
            "delta_value_usd": [3000.0],
        }
    )
    mock_compute_deltas.return_value = deltas_df
    mock_send_orders.return_value = [{"status": "filled", "id": "order-1"}]

    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "weight": [0.5, 0.5],
        }
    )

    result = rebalance_portfolio(target_weights)
    assert result["status"] == "completed"
    assert "order_deltas" in result
    mock_send_orders.assert_called_once()


@patch("app.services.execute.get_account")
def test_rebalance_portfolio_zero_account(mock_get_account: MagicMock):
    """Test rebalancing with zero account value."""
    mock_get_account.return_value = {"equity": "0.00"}

    target_weights = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "weight": [1.0],
        }
    )

    result = rebalance_portfolio(target_weights)
    assert result["status"] == "error"
    assert "zero" in result["error"].lower() or "negative" in result["error"].lower()
