"""
Order execution services for Quant50 (T006).

Implements Alpaca Paper Trading integration:
- Fetch current positions and account info
- Compute order deltas from target weights
- Send market/limit orders with validation
- Log all execution events
- Handle rate limits and errors gracefully
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests

from app.utils.config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""

    dry_run: bool = False  # If True, log but don't send orders
    order_type: str = "market"  # "market" or "limit"
    time_in_force: str = "day"  # "day", "gtc", "ioc", "fok"
    max_order_value_usd: float = 10000.0  # Safety limit per order
    rate_limit_delay: float = 0.2  # Seconds between API calls
    max_retries: int = 3
    retry_delay: float = 1.0


def _get_headers() -> Dict[str, str]:
    """Get Alpaca API headers."""
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID or "",
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY or "",
    }


def _handle_api_error(response: requests.Response, context: str) -> None:
    """Log and raise appropriate errors for API responses."""
    if response.status_code == 401:
        raise ValueError(f"Unauthorized (401): Check Alpaca API credentials. {context}")
    elif response.status_code == 429:
        raise ValueError(f"Rate limit exceeded (429): Too many requests. {context}")
    elif response.status_code >= 500:
        raise ValueError(
            f"Server error {response.status_code}: {response.text}. {context}"
        )
    elif response.status_code != 200:
        raise ValueError(
            f"API error {response.status_code}: {response.text}. {context}"
        )


def get_account() -> Dict[str, object]:
    """
    Get Alpaca account information.

    Returns
    -------
    dict
        Account info (buying_power, equity, cash, etc.)
    """
    url = f"{ALPACA_BASE_URL}/v2/account"
    headers = _get_headers()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        _handle_api_error(response, "get_account")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in get_account: {e}")
        raise


def get_current_positions() -> pd.DataFrame:
    """
    Get current portfolio positions from Alpaca.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, qty, market_value, avg_entry_price, current_price
    """
    url = f"{ALPACA_BASE_URL}/v2/positions"
    headers = _get_headers()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        _handle_api_error(response, "get_current_positions")

        data = response.json()
        if not data:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "qty",
                    "market_value",
                    "avg_entry_price",
                    "current_price",
                ]
            )

        df = pd.DataFrame(data)
        df = df[
            ["symbol", "qty", "market_value", "avg_entry_price", "current_price"]
        ].copy()
        df["qty"] = df["qty"].astype(float)
        df["market_value"] = df["market_value"].astype(float)
        df["avg_entry_price"] = df["avg_entry_price"].astype(float)
        df["current_price"] = df["current_price"].astype(float)

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in get_current_positions: {e}")
        raise


def get_recent_orders(limit: int = 50) -> pd.DataFrame:
    """
    Get recent orders from Alpaca.

    Parameters
    ----------
    limit : int, optional
        Maximum number of orders to retrieve (default: 50).

    Returns
    -------
    pd.DataFrame
        Columns: symbol, qty, side, status, filled_at, filled_avg_price
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    headers = _get_headers()
    params = {
        "status": "all",
        "limit": limit,
        "direction": "desc",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to get orders: {response.status_code}")
            return pd.DataFrame()

        orders = response.json()
        if not orders:
            return pd.DataFrame()

        df = pd.DataFrame(orders)
        # Select and rename relevant columns
        cols_map = {
            "symbol": "symbol",
            "qty": "qty",
            "side": "side",
            "status": "status",
            "filled_at": "filled_at",
            "filled_avg_price": "filled_avg_price",
        }
        available_cols = [c for c in cols_map.keys() if c in df.columns]
        df = df[available_cols].copy()
        df = df.rename(
            columns={c: cols_map[c] for c in available_cols if c in cols_map}
        )

        if "filled_at" in df.columns:
            df["filled_at"] = pd.to_datetime(df["filled_at"], errors="coerce")

        return df

    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return pd.DataFrame()


def compute_order_deltas(
    target_weights: pd.DataFrame,
    current_positions: pd.DataFrame,
    account_value: float,
) -> pd.DataFrame:
    """
    Compute order deltas (BUY/SELL/HOLD) between target portfolio weights
    and current positions.

    Parameters
    ----------
    target_weights : pd.DataFrame
        Columns: ['symbol', 'weight']
    current_positions : pd.DataFrame
        Columns: ['symbol', 'qty', 'current_price']
    account_value : float
        Total portfolio value (equity) in USD.

    Returns
    -------
    pd.DataFrame
        Columns:
        ['symbol', 'action', 'current_qty', 'target_qty',
         'target_weight', 'current_weight', 'delta_qty', 'delta_value_usd']
    """

    # Defensive copy
    target_weights = target_weights.copy()

    # --- Normalize weights only if they represent a full allocation ---
    total_weight = target_weights["weight"].sum()

    if 0.99 < total_weight < 1.01:
        # Normalize small numerical drifts (e.g., 0.9995 -> 1.0)
        target_weights["weight"] /= total_weight
    elif total_weight <= 0:
        logger.warning("Target weights sum to zero; no orders generated.")
        return pd.DataFrame(
            columns=[
                "symbol",
                "action",
                "current_qty",
                "target_qty",
                "target_weight",
                "current_weight",
                "delta_qty",
                "delta_value_usd",
            ]
        )
    else:
        # If sum < 1.0, keep as is (partial exposure allowed)
        logger.info(f"Partial portfolio allocation detected (Σw={total_weight:.3f})")

    # --- Enforce long-only: negative quantities clipped to zero ---
    current_positions = current_positions.copy()
    current_positions["qty"] = current_positions["qty"].clip(lower=0.0)

    # Merge current positions with target weights
    current_pos = current_positions.set_index("symbol")[["qty", "current_price"]].copy()
    current_pos["market_value"] = current_pos["qty"] * current_pos["current_price"]
    current_pos["current_weight"] = current_pos["market_value"] / account_value

    merged = (
        target_weights.set_index("symbol").join(current_pos, how="outer").fillna(0.0)
    )

    # --- Compute values and deltas ---
    merged["target_value"] = merged["weight"] * account_value
    merged["current_value"] = merged["qty"] * merged["current_price"]
    merged["delta_value_usd"] = merged["target_value"] - merged["current_value"]
    merged["delta_qty"] = merged["delta_value_usd"] / merged["current_price"]
    merged["delta_qty"] = merged["delta_qty"].fillna(0.0)

    # Compute target_qty explicitly (post-normalization)
    merged["target_qty"] = merged["target_value"] / merged["current_price"]
    merged["target_qty"] = merged["target_qty"].fillna(0.0)

    # --- Determine action (BUY / SELL / HOLD) ---
    def _action(row: pd.Series) -> str:
        if abs(row["delta_qty"]) < 1e-3:
            return "HOLD"
        elif row["delta_qty"] > 0:
            return "BUY"
        else:
            return "SELL"

    merged["action"] = merged.apply(_action, axis=1)

    # --- Round for realistic precision (2 decimals) ---
    merged["target_qty"] = merged["target_qty"].round(2)
    merged["delta_qty"] = merged["delta_qty"].round(2)
    merged["delta_value_usd"] = merged["delta_value_usd"].round(2)

    # --- Prepare final result ---
    result = merged.reset_index()[
        [
            "symbol",
            "action",
            "qty",
            "target_qty",
            "weight",
            "current_weight",
            "delta_qty",
            "delta_value_usd",
        ]
    ].rename(
        columns={
            "qty": "current_qty",
            "weight": "target_weight",
        }
    )

    # Filter out HOLD actions (no trade needed)
    result = result[result["action"] != "HOLD"].copy()

    # Sort by trade impact magnitude (largest first)
    result = result.sort_values("delta_value_usd", key=abs, ascending=False)

    logger.info(
        f"Order deltas computed — {len(result)} trades, "
        f"Σtarget_weight={target_weights['weight'].sum():.6f}"
    )

    return result


def send_order(
    symbol: str,
    qty: float,
    side: str,
    config: Optional[ExecutionConfig] = None,
) -> Dict[str, object]:
    """
    Send a single order to Alpaca.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "AAPL").
    qty : float
        Quantity to trade (positive for buy, negative for sell).
    side : str
        "buy" or "sell".
    config : ExecutionConfig, optional
        Execution configuration.

    Returns
    -------
    dict
        Order response from Alpaca API.
    """
    cfg = config or ExecutionConfig()
    side_lower = side.lower()
    if side_lower not in ["buy", "sell"]:
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

    # Round to 2 decimals (T005A requirement)
    qty_abs = round(abs(qty), 2)
    if qty_abs < 0.01:  # Minimum 0.01 shares (2 decimals)
        logger.warning(f"Skipping order for {symbol}: qty={qty} is too small")
        return {"status": "skipped", "reason": "qty_too_small"}

    url = f"{ALPACA_BASE_URL}/v2/orders"
    headers = _get_headers()
    headers["Content-Type"] = "application/json"

    payload = {
        "symbol": symbol,
        "qty": str(qty_abs),
        "side": side_lower,
        "type": cfg.order_type,
        "time_in_force": cfg.time_in_force,
    }

    if cfg.dry_run:
        logger.info(f"[DRY-RUN] Would send order: {payload}")
        return {
            "status": "dry_run",
            "symbol": symbol,
            "qty": qty_abs,
            "side": side_lower,
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        _handle_api_error(response, f"send_order({symbol})")

        order_data = response.json()
        logger.info(
            f"Order sent: {symbol} {side_lower} {qty_abs} "
            f"(order_id={order_data.get('id', 'N/A')})"
        )
        return order_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in send_order({symbol}): {e}")
        raise


def send_orders(
    order_deltas: pd.DataFrame,
    config: Optional[ExecutionConfig] = None,
) -> List[Dict[str, object]]:
    """
    Send multiple orders with rate limiting.

    Parameters
    ----------
    order_deltas : pd.DataFrame
        Output from compute_order_deltas().
    config : ExecutionConfig, optional
        Execution configuration.

    Returns
    -------
    list[dict]
        List of order responses from Alpaca API.
    """
    cfg = config or ExecutionConfig()
    results = []

    for _, row in order_deltas.iterrows():
        symbol = row["symbol"]
        delta_qty = row["delta_qty"]
        action = row["action"]

        if action == "HOLD":
            continue

        side = "buy" if action == "BUY" else "sell"

        try:
            order_result = send_order(symbol, delta_qty, side, config=cfg)
            results.append(order_result)

            # Rate limiting
            if not cfg.dry_run:
                time.sleep(cfg.rate_limit_delay)

        except Exception as e:
            logger.error(f"Failed to send order for {symbol}: {e}")
            results.append(
                {
                    "status": "error",
                    "symbol": symbol,
                    "error": str(e),
                }
            )

    return results


def validate_execution(
    target_weights: pd.DataFrame,
    tolerance: float = 0.05,
) -> Dict[str, object]:
    """
    Validate that executed positions match target weights (within tolerance).

    Parameters
    ----------
    target_weights : pd.DataFrame
        Expected weights (columns: symbol, weight).
    tolerance : float
        Maximum allowed weight deviation (default 0.05 = 5%).

    Returns
    -------
    dict
        Validation results with pass/fail status and details.
    """
    try:
        account = get_account()
        account_value = float(account.get("equity", 0.0))

        if account_value <= 0:
            return {
                "valid": False,
                "error": "Account value is zero or negative",
            }

        positions = get_current_positions()
        if positions.empty:
            # If no positions and target is also empty, that's valid
            if target_weights.empty or target_weights["weight"].sum() < 1e-6:
                return {"valid": True, "message": "No positions, target is empty"}
            else:
                return {
                    "valid": False,
                    "error": "Target has weights but no positions exist",
                }

        positions["qty"] = pd.to_numeric(positions["qty"], errors="coerce").fillna(0.0)
        positions["current_price"] = pd.to_numeric(
            positions["current_price"], errors="coerce"
        ).fillna(0.0)

        # Compute current weights
        positions["market_value"] = positions["qty"].abs() * positions["current_price"]
        positions["current_weight"] = positions["market_value"] / account_value

        # Merge with target
        target = target_weights.set_index("symbol")[["weight"]].copy()
        current = positions.set_index("symbol")[["current_weight"]].copy()

        merged = target.join(current, how="outer").fillna(0.0)

        # Check deviations
        merged["deviation"] = merged["current_weight"] - merged["weight"]
        max_deviation = merged["deviation"].abs().max()
        violations = merged[merged["deviation"].abs() > tolerance]

        valid = bool(max_deviation <= tolerance)

        result = {
            "valid": valid,
            "max_deviation": float(max_deviation),
            "tolerance": tolerance,
            "account_value": account_value,
            "violations": violations.to_dict("index") if not violations.empty else {},
        }

        if valid:
            logger.info(
                f"Execution validation passed: max_deviation={max_deviation:.4f} "
                f"<= tolerance={tolerance}"
            )
        else:
            logger.warning(
                f"Execution validation failed: max_deviation={max_deviation:.4f} "
                f"> tolerance={tolerance}. Violations: {len(violations)}"
            )

        return result

    except Exception as e:
        logger.error(f"Error in validate_execution: {e}")
        return {"valid": False, "error": str(e)}


def rebalance_portfolio(
    target_weights: pd.DataFrame,
    config: Optional[ExecutionConfig] = None,
) -> Dict[str, object]:
    """
    Complete portfolio rebalancing workflow.

    1. Get current positions and account info
    2. Compute order deltas
    3. Send orders
    4. Validate execution

    Parameters
    ----------
    target_weights : pd.DataFrame
        Target weights (columns: symbol, weight).
    config : ExecutionConfig, optional
        Execution configuration.

    Returns
    -------
    dict
        Execution summary with positions, orders, and validation.
    """
    cfg = config or ExecutionConfig()

    logger.info("Starting portfolio rebalancing...")

    try:
        # Get account and positions
        account = get_account()
        account_value = float(account.get("equity", 0.0))
        logger.info(f"Account equity: ${account_value:,.2f}")

        if account_value <= 0:
            raise ValueError("Account equity is zero or negative")

        positions = get_current_positions()
        logger.info(f"Current positions: {len(positions)} assets")

        # Compute order deltas
        deltas = compute_order_deltas(target_weights, positions, account_value)
        logger.info(f"Order deltas computed: {len(deltas)} trades needed")

        if deltas.empty:
            logger.info("No rebalancing needed")
            return {
                "status": "no_rebalance",
                "account_value": account_value,
                "positions": positions.to_dict("records"),
            }

        # Log deltas
        for _, row in deltas.iterrows():
            logger.info(
                f"  {row['action']} {row['symbol']}: "
                f"{row['delta_qty']:.4f} shares "
                f"(${row['delta_value_usd']:,.2f})"
            )

        # Send orders
        order_results = send_orders(deltas, config=cfg)
        logger.info(f"Orders sent: {len(order_results)}")

        # Wait a bit for orders to settle (if not dry run)
        if not cfg.dry_run:
            time.sleep(2.0)

        # Validate execution
        validation = validate_execution(target_weights, tolerance=0.05)

        return {
            "status": "completed",
            "account_value": account_value,
            "order_deltas": deltas.to_dict("records"),
            "orders_sent": order_results,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Error in rebalance_portfolio: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def cancel_all_orders() -> int:
    """
    Cancel all open orders. Used for Safety Halt.

    Returns
    -------
    int
        Number of orders cancelled (or attempted).
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    headers = _get_headers()

    try:
        # First, check if there are open orders
        response = requests.get(
            url, headers=headers, params={"status": "open"}, timeout=10
        )
        if response.status_code == 200:
            orders = response.json()
            if not orders:
                logger.info("No open orders to cancel.")
                return 0

        # Cancel all
        logger.warning(f"⚠️ Safety Halt: Cancelling {len(orders)} open orders...")
        response = requests.delete(url, headers=headers, timeout=10)

        if response.status_code == 207:  # Multi-status
            logger.info("Cancellation request sent (207).")
            return len(orders)
        elif response.status_code == 204:  # No content (success)
            logger.info("All orders cancelled successfully.")
            return len(orders)
        else:
            logger.error(
                f"Failed to cancel orders: {response.status_code} {response.text}"
            )
            return 0

    except Exception as e:
        logger.error(f"Error in cancel_all_orders: {e}")
        return 0
