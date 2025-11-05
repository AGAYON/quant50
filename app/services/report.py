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
import shutil
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fpdf import FPDF

from app.services.execute import get_account, get_current_positions
from app.utils.config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_BASE_URL,
    DUCKDB_PATH,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    """Remove or replace characters unsupported by Latin-1 (FPDF)."""
    if not isinstance(text, str):
        text = str(text)
    # Normalize to ASCII-compatible form
    text = unicodedata.normalize("NFKD", text)
    # Replace unsupported chars (like ✓, €, etc.)
    text = text.encode("latin-1", "replace").decode("latin-1")
    # Optional: replace placeholder "?" with something neutral
    return text.replace("?", "")


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

    # Calculate CAGR (Compound Annual Growth Rate)
    if len(equity) >= 2:
        days_diff = (
            equity_history["timestamp"].iloc[-1] - equity_history["timestamp"].iloc[0]
        ).days
        years = days_diff / 365.25 if days_diff > 0 else 1.0
        if years > 0 and starting_equity > 0:
            cagr = ((current_equity / starting_equity) ** (1.0 / years) - 1.0) * 100
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    return {
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "ytd_return": ytd_return,
        "cagr": cagr,
        "current_equity": current_equity,
        "starting_equity": starting_equity,
    }


def calculate_risk_metrics(
    equity_history: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate risk metrics: drawdown, volatility, turnover, Sharpe, Sortino, VaR, CVaR.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Must have 'timestamp' and 'equity' columns.
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.0).

    Returns
    -------
    dict
        {
            'max_drawdown': float (as percentage),
            'volatility': float (annualized, as percentage),
            'turnover': float (estimated from equity changes),
            'sharpe_ratio': float (annualized),
            'sortino_ratio': float (annualized),
            'var_95': float (Value-at-Risk 95%),
            'cvar_95': float (Conditional VaR 95%),
            'rolling_volatility_30d': float (30-day rolling, annualized %)
        }
    """
    if equity_history.empty or "equity" not in equity_history.columns:
        return {
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "rolling_volatility_30d": 0.0,
        }

    equity_history = equity_history.sort_values("timestamp")
    equity = equity_history["equity"].dropna()

    if len(equity) < 2:
        return {
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "rolling_volatility_30d": 0.0,
        }

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Calculate returns
    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        return {
            "max_drawdown": max_drawdown,
            "volatility": 0.0,
            "turnover": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "rolling_volatility_30d": 0.0,
        }

    # Annualized volatility
    vol_annualized = returns.std() * (252**0.5) * 100

    # Rolling 30-day volatility
    rolling_vol = returns.rolling(window=min(30, len(returns))).std()
    rolling_vol_30d = (
        rolling_vol.iloc[-1] * (252**0.5) * 100 if len(rolling_vol) > 0 else 0.0
    )

    # Sharpe ratio (annualized)
    mean_return = returns.mean() * 252  # Annualized mean return
    sharpe = (
        (mean_return - risk_free_rate) / (returns.std() * (252**0.5))
        if returns.std() > 0
        else 0.0
    )

    # Sortino ratio (annualized, downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = (
        downside_returns.std() * (252**0.5) if len(downside_returns) > 0 else 0.0
    )
    sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    # Value-at-Risk (VaR 95%) and Conditional VaR (CVaR 95%)
    var_95 = np.percentile(returns, 5) * 100  # 5th percentile (negative)
    cvar_95 = (
        returns[returns <= np.percentile(returns, 5)].mean() * 100
        if len(returns[returns <= np.percentile(returns, 5)]) > 0
        else 0.0
    )

    # Estimate turnover (simplified: average daily equity change %)
    daily_changes = equity.pct_change().abs().dropna()
    turnover = daily_changes.mean() * 100 if len(daily_changes) > 0 else 0.0

    return {
        "max_drawdown": max_drawdown,
        "volatility": vol_annualized,
        "turnover": turnover,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "rolling_volatility_30d": rolling_vol_30d,
    }


def calculate_diversification_index(
    positions: pd.DataFrame,
) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI) for portfolio diversification.

    HHI ranges from 0 (perfectly diversified) to 1 (concentrated).
    Lower HHI = better diversification.

    Parameters
    ----------
    positions : pd.DataFrame
        Must have 'market_value' column.

    Returns
    -------
    float
        HHI value (0-1 range).
    """
    if positions.empty or "market_value" not in positions.columns:
        return 1.0  # Maximum concentration if no positions

    total_value = positions["market_value"].sum()
    if total_value == 0:
        return 1.0

    weights = positions["market_value"] / total_value
    hhi = (weights**2).sum()
    return float(hhi)


def calculate_order_metrics(
    orders: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate order execution metrics: fill rate, average holding period.

    Parameters
    ----------
    orders : pd.DataFrame
        Must have 'status' and optionally 'filled_at' columns.

    Returns
    -------
    dict
        {
            'fill_rate': float (0-1),
            'avg_holding_period_days': float (if data available)
        }
    """
    if orders.empty:
        return {"fill_rate": 0.0, "avg_holding_period_days": 0.0}

    # Fill rate: % of orders that are filled
    if "status" in orders.columns:
        filled_count = (orders["status"] == "filled").sum()
        fill_rate = filled_count / len(orders) if len(orders) > 0 else 0.0
    else:
        fill_rate = 0.0

    # Average holding period (requires entry/exit data, simplified for now)
    avg_holding_period = 0.0  # Placeholder - would need position tracking

    return {
        "fill_rate": fill_rate,
        "avg_holding_period_days": avg_holding_period,
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

    # Get historical equity (with fallback to cache)
    equity_history, data_source_status = get_account_history_with_fallback(
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=report_date,
        use_cache=True,
    )

    # Calculate metrics
    pnl_metrics = calculate_pnl_metrics(equity_history)
    risk_metrics = calculate_risk_metrics(equity_history)

    # Calculate advanced metrics
    diversification_index = calculate_diversification_index(positions)

    # Get top holdings
    holdings_data = get_top_holdings(positions, n=10, symbol_to_sector=symbol_to_sector)

    # Get recent orders
    orders = get_recent_orders(limit=50)
    order_metrics = calculate_order_metrics(orders)

    report = {
        "report_date": report_date,
        "account": {
            "equity": (
                float(account.get("equity", 0.0))
                if isinstance(account.get("equity"), str)
                else account.get("equity", 0.0)
            ),
            "cash": (
                float(account.get("cash", 0.0))
                if isinstance(account.get("cash"), str)
                else account.get("cash", 0.0)
            ),
            "buying_power": (
                float(account.get("buying_power", 0.0))
                if isinstance(account.get("buying_power"), str)
                else account.get("buying_power", 0.0)
            ),
        },
        "pnl_metrics": pnl_metrics,
        "risk_metrics": risk_metrics,
        "diversification_index": diversification_index,
        "order_metrics": order_metrics,
        "top_holdings": holdings_data["top_holdings"].to_dict("records"),
        "sector_weights": holdings_data["sector_weights"].to_dict("records"),
        "positions_count": len(positions),
        "recent_orders": orders.to_dict("records") if not orders.empty else [],
        "orders_count": len(orders),
        "equity_history": equity_history,  # Include for chart generation
        "data_source_status": data_source_status,  # Track API vs cache usage
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
        self.cell(0, 10, sanitize_text("Quant50 Daily Report"), 0, 1, "C")
        self.ln(5)

    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, sanitize_text(f"Page {self.page_no()}"), 0, 0, "C")

    def section_title(self, title: str):
        """Add a section title."""
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, sanitize_text(title), 0, 1)
        self.ln(2)

    def metric_row(self, label: str, value: str):
        """Add a metric row."""
        self.set_font("Arial", "", 10)
        self.cell(90, 8, sanitize_text(label), 0, 0)
        self.set_font("Arial", "B", 10)
        self.cell(0, 8, sanitize_text(value), 0, 1)


def create_pdf_report(
    report_data: Dict[str, object],
    output_path: Optional[str] = None,
    include_charts: bool = True,
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

    # Generate charts if requested
    chart_paths = {}
    chart_dir = None
    if include_charts:
        try:
            from app.services.report_charts import generate_all_charts

            chart_dir = os.path.join(REPORTS_DIR, "tmp_charts")
            equity_history = report_data.get("equity_history", pd.DataFrame())
            sector_weights_df = pd.DataFrame(report_data.get("sector_weights", []))
            top_holdings_df = pd.DataFrame(report_data.get("top_holdings", []))
            chart_paths = generate_all_charts(
                equity_history, sector_weights_df, top_holdings_df, chart_dir
            )
        except Exception as e:
            logger.warning(f"Failed to generate charts: {e}")
            include_charts = False

    pdf = PDFReport()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Quant50 Daily Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, sanitize_text(f"Date: {report_data['report_date']}"), 0, 1, "C")
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

    # Risk Metrics with color coding
    pdf.section_title("Risk Metrics")
    risk = report_data.get("risk_metrics", {})

    # Max Drawdown (red if < -10%)
    max_dd = risk.get("max_drawdown", 0)
    if max_dd < -10.0:
        pdf.set_text_color(255, 0, 0)  # Red color
    pdf.metric_row("Max Drawdown", f"{max_dd:.2f}%")
    pdf.set_text_color(0, 0, 0)  # Reset to black

    # Volatility (orange if > 25%)
    vol = risk.get("volatility", 0)
    if vol > 25.0:
        pdf.set_text_color(255, 165, 0)  # Orange color
    pdf.metric_row("Volatility (Annualized)", f"{vol:.2f}%")
    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.metric_row(
        "Rolling Volatility (30D)", f"{risk.get('rolling_volatility_30d', 0):.2f}%"
    )
    pdf.metric_row("Turnover", f"{risk.get('turnover', 0):.2f}%")
    pdf.metric_row("Sharpe Ratio", f"{risk.get('sharpe_ratio', 0):.2f}")
    pdf.metric_row("Sortino Ratio", f"{risk.get('sortino_ratio', 0):.2f}")
    pdf.metric_row("VaR (95%)", f"{risk.get('var_95', 0):.2f}%")
    pdf.metric_row("CVaR (95%)", f"{risk.get('cvar_95', 0):.2f}%")
    pdf.ln(5)

    # Advanced Metrics
    pdf.section_title("Advanced Metrics")
    pdf.metric_row(
        "Diversification Index (HHI)",
        f"{report_data.get('diversification_index', 1.0):.4f}",
    )
    order_metrics = report_data.get("order_metrics", {})
    pdf.metric_row("Order Fill Rate", f"{order_metrics.get('fill_rate', 0) * 100:.2f}%")
    pdf.ln(5)

    # Embed charts if available
    if include_charts and chart_paths:
        # Equity Curve
        if "equity_curve" in chart_paths:
            pdf.section_title("Equity Curve")
            pdf.image(chart_paths["equity_curve"], x=10, y=pdf.get_y(), w=190, h=100)
            pdf.ln(110)

        # Drawdown Curve
        if "drawdown" in chart_paths:
            pdf.section_title("Drawdown Curve")
            pdf.image(chart_paths["drawdown"], x=10, y=pdf.get_y(), w=190, h=100)
            pdf.ln(110)

        # Sector Weights
        if "sector_weights" in chart_paths:
            pdf.section_title("Sector Weight Distribution")
            pdf.image(chart_paths["sector_weights"], x=10, y=pdf.get_y(), w=190, h=100)
            pdf.ln(110)

        # Top Holdings
        if "top_holdings" in chart_paths:
            pdf.section_title("Top Holdings")
            pdf.image(chart_paths["top_holdings"], x=10, y=pdf.get_y(), w=190, h=100)
            pdf.ln(110)

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

    # Notes Section
    pdf.add_page()
    pdf.section_title("Notes")
    pdf.set_font("Arial", "", 10)

    # Data source status
    data_source_status = report_data.get("data_source_status", "unknown")
    if data_source_status == "api_success":
        pdf.cell(0, 8, "Successfully connected to Alpaca API for account history", 0, 1)
    elif data_source_status == "cache_used":
        pdf.set_text_color(255, 165, 0)  # Orange color
        pdf.cell(
            0,
            8,
            "Alpaca API connection failed; used DuckDB cache for account history",
            0,
            1,
        )
        pdf.set_text_color(0, 0, 0)  # Reset to black
    elif data_source_status == "api_failed":
        pdf.set_text_color(255, 0, 0)  # Red color
        pdf.cell(0, 8, "Alpaca API connection failed; no cache available", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Reset to black

    pdf.ln(3)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(
        0,
        8,
        f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        0,
        1,
    )
    pdf.cell(0, 8, "Quant50 Portfolio Management System", 0, 1)

    pdf.output(output_path)
    logger.info(f"PDF report saved to {output_path}")

    # Clean up temporary chart files
    if chart_dir and os.path.exists(chart_dir):
        try:
            shutil.rmtree(chart_dir)
            logger.info(f"Cleaned up temporary charts directory: {chart_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up chart directory: {e}")

    return output_path


def get_account_history_with_fallback(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Get account history with DuckDB fallback cache.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
    use_cache : bool, optional
        If True, try DuckDB cache if API fails (default: True).

    Returns
    -------
    tuple[pd.DataFrame, str]
        Equity history DataFrame and status message:
        - "api_success": Successfully retrieved from Alpaca API
        - "cache_used": Fallback to DuckDB cache used
        - "api_failed": API failed and no cache available
    """
    try:
        # Try API first
        result = get_account_history(start_date=start_date, end_date=end_date)
        return result, "api_success"
    except Exception as e:
        logger.warning(f"Failed to get account history from API: {e}")
        if use_cache:
            # Fallback to DuckDB cache if available
            try:
                import duckdb

                con = duckdb.connect(DUCKDB_PATH, read_only=True)
                query = """
                    SELECT DISTINCT timestamp, equity, cash, buying_power
                    FROM account_history
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """
                result = con.execute(
                    query,
                    [
                        start_date
                        or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        end_date or datetime.now().strftime("%Y-%m-%d"),
                    ],
                ).fetchdf()
                con.close()
                if not result.empty:
                    logger.info("Retrieved account history from DuckDB cache")
                    return result, "cache_used"
            except Exception as cache_error:
                logger.error(f"Cache fallback also failed: {cache_error}")

        # Return empty DataFrame if all fails
        return (
            pd.DataFrame(columns=["timestamp", "equity", "cash", "buying_power"]),
            "api_failed",
        )
