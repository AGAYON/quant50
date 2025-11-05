"""
Daily PDF report generation for Quant50 (T007).

Generates comprehensive daily reports with:
- PnL (daily, cumulative, YTD)
- Risk metrics (drawdown, volatility, turnover)
- Portfolio composition (top holdings, sector weights)
- Executed orders summary
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import requests
from fpdf import FPDF

from app.services.execute import get_account, get_current_positions
from app.utils.config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_BASE_URL,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def _get_headers() -> Dict[str, str]:
    """Get Alpaca API headers."""
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID or "",
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY or "",
    }


def get_account_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get historical account equity from Alpaca API.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 30 days ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, equity, cash, buying_power
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    url = f"{ALPACA_BASE_URL}/v2/account/portfolio/history"
    headers = _get_headers()
    params = {
        "start": start_date,
        "end": end_date,
        "period": "1D",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to get account history: {response.status_code}")
            return pd.DataFrame(columns=["timestamp", "equity", "cash", "buying_power"])

        data = response.json()
        if "equity" not in data:
            return pd.DataFrame(columns=["timestamp", "equity", "cash", "buying_power"])

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data.get("timestamp", [])),
                "equity": data.get("equity", []),
                "cash": data.get("cash", []),
                "buying_power": data.get("buying_power", []),
            }
        )

        return df

    except Exception as e:
        logger.error(f"Error fetching account history: {e}")
        return pd.DataFrame(columns=["timestamp", "equity", "cash", "buying_power"])


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


def calculate_pnl_metrics(
    equity_history: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate PnL metrics from equity history.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Must have 'timestamp' and 'equity' columns.

    Returns
    -------
    dict
        {
            'daily_pnl': float,
            'cumulative_pnl': float,
            'ytd_return': float,
            'current_equity': float,
            'starting_equity': float
        }
    """
    if equity_history.empty or "equity" not in equity_history.columns:
        return {
            "daily_pnl": 0.0,
            "cumulative_pnl": 0.0,
            "ytd_return": 0.0,
            "current_equity": 0.0,
            "starting_equity": 0.0,
        }

    equity_history = equity_history.sort_values("timestamp")
    equity = equity_history["equity"].dropna()

    if len(equity) < 2:
        current_equity = equity.iloc[-1] if len(equity) > 0 else 0.0
        return {
            "daily_pnl": 0.0,
            "cumulative_pnl": 0.0,
            "ytd_return": 0.0,
            "current_equity": current_equity,
            "starting_equity": current_equity,
        }

    current_equity = equity.iloc[-1]
    starting_equity = equity.iloc[0]
    daily_pnl = equity.iloc[-1] - equity.iloc[-2] if len(equity) >= 2 else 0.0
    cumulative_pnl = current_equity - starting_equity

    # YTD return
    current_year = datetime.now().year
    ytd_start = equity_history[
        equity_history["timestamp"] >= pd.Timestamp(f"{current_year}-01-01")
    ]
    if not ytd_start.empty:
        ytd_start_equity = ytd_start["equity"].iloc[0]
        ytd_return = (
            ((current_equity - ytd_start_equity) / ytd_start_equity * 100)
            if ytd_start_equity > 0
            else 0.0
        )
    else:
        ytd_return = 0.0

    return {
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "ytd_return": ytd_return,
        "current_equity": current_equity,
        "starting_equity": starting_equity,
    }


def calculate_risk_metrics(
    equity_history: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate risk metrics: drawdown, volatility, turnover.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Must have 'timestamp' and 'equity' columns.

    Returns
    -------
    dict
        {
            'max_drawdown': float (as percentage),
            'volatility': float (annualized, as percentage),
            'turnover': float (estimated from equity changes)
        }
    """
    if equity_history.empty or "equity" not in equity_history.columns:
        return {
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
        }

    equity_history = equity_history.sort_values("timestamp")
    equity = equity_history["equity"].dropna()

    if len(equity) < 2:
        return {
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
        }

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Calculate volatility (annualized)
    returns = equity.pct_change().dropna()
    if len(returns) > 1:
        volatility = returns.std() * (252**0.5) * 100  # Annualized
    else:
        volatility = 0.0

    # Estimate turnover (simplified: average daily equity change %)
    daily_changes = equity.pct_change().abs().dropna()
    turnover = daily_changes.mean() * 100 if len(daily_changes) > 0 else 0.0

    return {
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "turnover": turnover,
    }


def get_top_holdings(
    positions: pd.DataFrame,
    n: int = 10,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Get top holdings and sector weights.

    Parameters
    ----------
    positions : pd.DataFrame
        Current positions from get_current_positions().
    n : int, optional
        Number of top holdings to return (default: 10).
    symbol_to_sector : dict, optional
        Mapping symbol → sector.

    Returns
    -------
    dict
        {
            'top_holdings': pd.DataFrame with columns [symbol, market_value, weight],
            'sector_weights': pd.DataFrame with columns [sector, weight]
            (if sector mapping provided)
        }
    """
    if positions.empty:
        return {
            "top_holdings": pd.DataFrame(columns=["symbol", "market_value", "weight"]),
            "sector_weights": pd.DataFrame(columns=["sector", "weight"]),
        }

    # Calculate total portfolio value
    total_value = positions["market_value"].sum()

    if total_value == 0:
        return {
            "top_holdings": pd.DataFrame(columns=["symbol", "market_value", "weight"]),
            "sector_weights": pd.DataFrame(columns=["sector", "weight"]),
        }

    # Top holdings
    positions["weight"] = positions["market_value"] / total_value * 100
    top_holdings = (
        positions[["symbol", "market_value", "weight"]]
        .sort_values("market_value", ascending=False)
        .head(n)
        .copy()
    )

    # Sector weights
    sector_weights = pd.DataFrame(columns=["sector", "weight"])
    if symbol_to_sector and not positions.empty:
        positions_with_sector = positions.copy()
        positions_with_sector["sector"] = positions_with_sector["symbol"].map(
            symbol_to_sector
        )
        positions_with_sector["sector"] = positions_with_sector["sector"].fillna("UNK")
        sector_weights = (
            positions_with_sector.groupby("sector")["market_value"].sum().reset_index()
        )
        sector_weights["weight"] = (
            sector_weights["market_value"] / total_value * 100
        ).round(2)
        sector_weights = sector_weights.sort_values("weight", ascending=False)[
            ["sector", "weight"]
        ]

    return {
        "top_holdings": top_holdings,
        "sector_weights": sector_weights,
    }


def generate_daily_report(
    report_date: Optional[str] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Generate daily report with all KPIs.

    Parameters
    ----------
    report_date : str, optional
        Date in YYYY-MM-DD format. Defaults to today.
    symbol_to_sector : dict, optional
        Mapping symbol → sector for sector analysis.

    Returns
    -------
    dict
        Report data with KPIs, positions, orders, etc.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Generating daily report for {report_date}...")

    # Get account and positions
    account = get_account()
    positions = get_current_positions()

    # Get historical equity
    equity_history = get_account_history(
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=report_date,
    )

    # Calculate metrics
    pnl_metrics = calculate_pnl_metrics(equity_history)
    risk_metrics = calculate_risk_metrics(equity_history)

    # Get top holdings
    holdings_data = get_top_holdings(positions, n=10, symbol_to_sector=symbol_to_sector)

    # Get recent orders
    orders = get_recent_orders(limit=50)

    report = {
        "report_date": report_date,
        "account": {
            "equity": account.get("equity", 0.0),
            "cash": account.get("cash", 0.0),
            "buying_power": account.get("buying_power", 0.0),
        },
        "pnl_metrics": pnl_metrics,
        "risk_metrics": risk_metrics,
        "top_holdings": holdings_data["top_holdings"].to_dict("records"),
        "sector_weights": holdings_data["sector_weights"].to_dict("records"),
        "positions_count": len(positions),
        "recent_orders": orders.to_dict("records") if not orders.empty else [],
        "orders_count": len(orders),
    }

    logger.info(
        f"Daily report generated: {len(positions)} positions, {len(orders)} orders"
    )
    return report


class PDFReport(FPDF):
    """Custom PDF class for Quant50 reports."""

    def header(self):
        """Add header to each page."""
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Quant50 Daily Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def section_title(self, title: str):
        """Add a section title."""
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)

    def metric_row(self, label: str, value: str):
        """Add a metric row."""
        self.set_font("Arial", "", 10)
        self.cell(90, 8, label, 0, 0)
        self.set_font("Arial", "B", 10)
        self.cell(0, 8, value, 0, 1)


def create_pdf_report(
    report_data: Dict[str, object],
    output_path: Optional[str] = None,
) -> str:
    """
    Create PDF report from report data.

    Parameters
    ----------
    report_data : dict
        Output from generate_daily_report().
    output_path : str, optional
        Full path for output PDF. If None, uses REPORTS_DIR/diario_YYYYMMDD.pdf.

    Returns
    -------
    str
        Path to generated PDF file.
    """
    if output_path is None:
        report_date = report_data.get(
            "report_date", datetime.now().strftime("%Y-%m-%d")
        )
        date_str = datetime.strptime(report_date, "%Y-%m-%d").strftime("%Y%m%d")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(REPORTS_DIR, f"diario_{date_str}.pdf")

    pdf = PDFReport()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Daily Report - {report_data['report_date']}", 0, 1, "C")
    pdf.ln(5)

    # Account Summary
    pdf.section_title("Account Summary")
    account = report_data.get("account", {})
    pdf.metric_row("Equity", f"${float(account.get('equity', 0)):,.2f}")
    pdf.metric_row("Cash", f"${float(account.get('cash', 0)):,.2f}")
    pdf.metric_row("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
    pdf.metric_row("Positions Count", str(report_data.get("positions_count", 0)))
    pdf.ln(5)

    # PnL Metrics
    pdf.section_title("PnL Metrics")
    pnl = report_data.get("pnl_metrics", {})
    pdf.metric_row("Daily PnL", f"${pnl.get('daily_pnl', 0):,.2f}")
    pdf.metric_row("Cumulative PnL", f"${pnl.get('cumulative_pnl', 0):,.2f}")
    pdf.metric_row("YTD Return", f"{pnl.get('ytd_return', 0):.2f}%")
    pdf.metric_row("Starting Equity", f"${pnl.get('starting_equity', 0):,.2f}")
    pdf.ln(5)

    # Risk Metrics
    pdf.section_title("Risk Metrics")
    risk = report_data.get("risk_metrics", {})
    pdf.metric_row("Max Drawdown", f"{risk.get('max_drawdown', 0):.2f}%")
    pdf.metric_row("Volatility (Annualized)", f"{risk.get('volatility', 0):.2f}%")
    pdf.metric_row("Turnover", f"{risk.get('turnover', 0):.2f}%")
    pdf.ln(5)

    # Top Holdings
    pdf.section_title("Top Holdings")
    top_holdings = report_data.get("top_holdings", [])
    if top_holdings:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Symbol", 1, 0, "C")
        pdf.cell(60, 8, "Market Value", 1, 0, "C")
        pdf.cell(60, 8, "Weight %", 1, 1, "C")
        pdf.set_font("Arial", "", 9)
        for holding in top_holdings[:10]:
            pdf.cell(60, 8, str(holding.get("symbol", "")), 1, 0)
            pdf.cell(60, 8, f"${holding.get('market_value', 0):,.2f}", 1, 0)
            pdf.cell(60, 8, f"{holding.get('weight', 0):.2f}%", 1, 1)
    else:
        pdf.cell(0, 8, "No positions", 0, 1)
    pdf.ln(5)

    # Sector Weights
    sector_weights = report_data.get("sector_weights", [])
    if sector_weights:
        pdf.section_title("Sector Weights")
        pdf.set_font("Arial", "B", 10)
        pdf.cell(90, 8, "Sector", 1, 0, "C")
        pdf.cell(90, 8, "Weight %", 1, 1, "C")
        pdf.set_font("Arial", "", 9)
        for sector in sector_weights:
            pdf.cell(90, 8, str(sector.get("sector", "")), 1, 0)
            pdf.cell(90, 8, f"{sector.get('weight', 0):.2f}%", 1, 1)
        pdf.ln(5)

    # Recent Orders
    pdf.section_title("Recent Orders")
    orders = report_data.get("recent_orders", [])
    if orders:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(40, 8, "Symbol", 1, 0, "C")
        pdf.cell(40, 8, "Side", 1, 0, "C")
        pdf.cell(40, 8, "Qty", 1, 0, "C")
        pdf.cell(40, 8, "Status", 1, 0, "C")
        pdf.cell(30, 8, "Price", 1, 1, "C")
        pdf.set_font("Arial", "", 8)
        for order in orders[:20]:  # Show last 20 orders
            pdf.cell(40, 8, str(order.get("symbol", ""))[:8], 1, 0)
            pdf.cell(40, 8, str(order.get("side", "")), 1, 0)
            pdf.cell(40, 8, str(order.get("qty", "")), 1, 0)
            pdf.cell(40, 8, str(order.get("status", ""))[:10], 1, 0)
            price = order.get("filled_avg_price", "")
            pdf.cell(30, 8, str(price)[:8] if price else "N/A", 1, 1)
    else:
        pdf.cell(0, 8, "No recent orders", 0, 1)

    pdf.output(output_path)
    logger.info(f"PDF report saved to {output_path}")
    return output_path
