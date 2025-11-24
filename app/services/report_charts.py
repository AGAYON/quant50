"""
Chart generation for Quant50 daily reports (T007B).

Generates matplotlib visualizations for equity curves, drawdowns,
sector weights, and top holdings.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)

# Chart style configuration
CHART_WIDTH = 8.0
CHART_HEIGHT = 5.0
DPI = 100
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#06A77D",
    "warning": "#F18F01",
    "danger": "#C73E1D",
    "gray": "#6C757D",
}


def _setup_chart_style(figsize: Tuple[float, float] = (CHART_WIDTH, CHART_HEIGHT)):
    """Setup matplotlib style for professional charts."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        # Fallback if seaborn style not available
        plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_equity_curve(
    equity_history: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Plot equity curve over time.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Must have 'timestamp' and 'equity' columns.
    output_path : str
        Full path for output PNG file.

    Returns
    -------
    str
        Path to saved chart.
    """
    if equity_history.empty or "equity" not in equity_history.columns:
        logger.warning("Empty equity history, creating placeholder chart")
        fig, ax = _setup_chart_style()
        ax.text(0.5, 0.5, "No equity data available", ha="center", va="center")
        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
        plt.close()
        return output_path

    equity_history = equity_history.sort_values("timestamp")
    equity = equity_history["equity"].dropna()

    fig, ax = _setup_chart_style()
    ax.plot(equity_history["timestamp"], equity, linewidth=2, color=COLORS["primary"])
    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity ($)", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
    plt.close()

    logger.info(f"Equity curve chart saved to {output_path}")
    return output_path


def plot_drawdown_curve(
    equity_history: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Plot drawdown curve over time.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Must have 'timestamp' and 'equity' columns.
    output_path : str
        Full path for output PNG file.

    Returns
    -------
    str
        Path to saved chart.
    """
    if equity_history.empty or "equity" not in equity_history.columns:
        logger.warning("Empty equity history, creating placeholder chart")
        fig, ax = _setup_chart_style()
        ax.text(0.5, 0.5, "No equity data available", ha="center", va="center")
        ax.set_title("Drawdown Curve", fontsize=14, fontweight="bold")
        plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
        plt.close()
        return output_path

    equity_history = equity_history.sort_values("timestamp")
    equity = equity_history["equity"].dropna()

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100

    fig, ax = _setup_chart_style()
    ax.fill_between(
        equity_history["timestamp"],
        drawdown,
        0,
        alpha=0.3,
        color=COLORS["danger"],
        label="Drawdown",
    )
    ax.plot(
        equity_history["timestamp"],
        drawdown,
        linewidth=1.5,
        color=COLORS["danger"],
    )
    ax.set_title("Drawdown Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
    plt.close()

    logger.info(f"Drawdown curve chart saved to {output_path}")
    return output_path


def plot_sector_weights(
    sector_weights: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Plot sector weight distribution as horizontal bar chart.

    Parameters
    ----------
    sector_weights : pd.DataFrame
        Must have 'sector' and 'weight' columns.
    output_path : str
        Full path for output PNG file.

    Returns
    -------
    str
        Path to saved chart.
    """
    if sector_weights.empty or "weight" not in sector_weights.columns:
        logger.warning("Empty sector weights, creating placeholder chart")
        fig, ax = _setup_chart_style()
        ax.text(0.5, 0.5, "No sector data available", ha="center", va="center")
        ax.set_title("Sector Weights", fontsize=14, fontweight="bold")
        plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
        plt.close()
        return output_path

    sector_weights = sector_weights.sort_values("weight", ascending=True)

    fig, ax = _setup_chart_style((CHART_WIDTH, max(4.0, len(sector_weights) * 0.4)))
    colors_list = [COLORS["primary"]] * len(sector_weights)
    ax.barh(
        sector_weights["sector"],
        sector_weights["weight"],
        color=colors_list,
        alpha=0.7,
    )
    ax.set_title("Sector Weight Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weight (%)", fontsize=11)
    ax.set_ylabel("Sector", fontsize=11)
    ax.axvline(
        x=50.0,
        color=COLORS["warning"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="50% Cap",
    )
    if len(sector_weights) > 0:
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
    plt.close()

    logger.info(f"Sector weights chart saved to {output_path}")
    return output_path


def plot_top_holdings(
    top_holdings: pd.DataFrame,
    output_path: str,
    n: int = 10,
) -> str:
    """
    Plot top holdings as horizontal bar chart.

    Parameters
    ----------
    top_holdings : pd.DataFrame
        Must have 'symbol' and 'weight' columns.
    output_path : str
        Full path for output PNG file.
    n : int, optional
        Number of holdings to show (default: 10).

    Returns
    -------
    str
        Path to saved chart.
    """
    if top_holdings.empty or "weight" not in top_holdings.columns:
        logger.warning("Empty top holdings, creating placeholder chart")
        fig, ax = _setup_chart_style()
        ax.text(0.5, 0.5, "No holdings data available", ha="center", va="center")
        ax.set_title("Top Holdings", fontsize=14, fontweight="bold")
        plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
        plt.close()
        return output_path

    top_n = top_holdings.head(n).sort_values("weight", ascending=True)

    fig, ax = _setup_chart_style((CHART_WIDTH, max(4.0, len(top_n) * 0.3)))
    colors_list = [COLORS["secondary"]] * len(top_n)
    ax.barh(top_n["symbol"], top_n["weight"], color=colors_list, alpha=0.7)
    ax.set_title(f"Top {len(top_n)} Holdings by Weight", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weight (%)", fontsize=11)
    ax.set_ylabel("Symbol", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=DPI)
    plt.close()

    logger.info(f"Top holdings chart saved to {output_path}")
    return output_path


def generate_all_charts(
    equity_history: pd.DataFrame,
    sector_weights: pd.DataFrame,
    top_holdings: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate all charts for the daily report.

    Parameters
    ----------
    equity_history : pd.DataFrame
        Equity history data.
    sector_weights : pd.DataFrame
        Sector weights data.
    top_holdings : pd.DataFrame
        Top holdings data.
    output_dir : str
        Directory to save chart PNGs.

    Returns
    -------
    dict
        Mapping of chart names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    chart_paths = {}

    # Equity curve
    equity_path = os.path.join(output_dir, "equity_curve.png")
    plot_equity_curve(equity_history, equity_path)
    chart_paths["equity_curve"] = equity_path

    # Drawdown curve
    drawdown_path = os.path.join(output_dir, "drawdown_curve.png")
    plot_drawdown_curve(equity_history, drawdown_path)
    chart_paths["drawdown"] = drawdown_path

    # Sector weights
    if not sector_weights.empty:
        sector_path = os.path.join(output_dir, "sector_weights.png")
        plot_sector_weights(sector_weights, sector_path)
        chart_paths["sector_weights"] = sector_path

    # Top holdings
    if not top_holdings.empty:
        holdings_path = os.path.join(output_dir, "top_holdings.png")
        plot_top_holdings(top_holdings, holdings_path, n=10)
        chart_paths["top_holdings"] = holdings_path

    return chart_paths
