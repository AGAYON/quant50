"""
app/services/data.py
~~~~~~~~~~~~~~~~~~~~

Module responsible for financial data ingestion using yfinance API.

Main functions:
- fetch_stock_data: downloads historical OHLCV prices for a given symbol.
- fetch_multiple_symbols: downloads multiple symbols and concatenates results.
- store_to_duckdb: saves DataFrame to DuckDB database for persistence.

Best practices:
- Uses yfinance for reliable financial data access
- Robust error handling and logging
- DuckDB integration for efficient data storage
- Returns clean DataFrames ready for analysis or modeling

Author: Andrés Gayón
Project: quant50
"""

# app/services/data.py

import os
import time
import logging
from datetime import datetime

import yfinance as yf
import pandas as pd
import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.utils.config import DATABASE_PATH


# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Data Fetching Functions
# -------------------------------------------------------------------------
def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock prices for a given symbol using yfinance.

    This function builds a custom HTTP session with retry logic and 
    browser-like headers to avoid rate limiting (HTTP 429) responses 
    from Yahoo Finance. It also ensures compatibility by disabling curl_cffi.

    Args:
        symbol (str): Asset ticker (e.g., 'AAPL', 'SPY')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with columns [Date, Open, High, Low, Close,
                     Adj Close, Volume, Symbol]
    """

    # Enforce yfinance to use requests instead of curl_cffi (avoids SSL errors)
    os.environ["YFINANCE_USE_CURL_CFFI"] = "false"

    # Build custom session to mimic browser and add retry strategy
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    try:
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        # Reattempt once if DataFrame returned empty
        if df.empty:
            logger.warning(f"⚠️ No se obtuvieron datos para {symbol}, reintentando...")
            time.sleep(2)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            logger.warning(f"⚠️ No se obtuvieron datos para {symbol}.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df["Symbol"] = symbol
        logger.info(f"✅ Datos descargados correctamente para {symbol}.")
        return df

    except Exception as e:
        logger.error(f"❌ Error descargando {symbol}: {e}")
        return pd.DataFrame()


def fetch_multiple_symbols(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download multiple assets and concatenate the results.

    Args:
        symbols (list[str]): List of tickers to download
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Combined DataFrame with all symbols
    """
    all_data = []

    for sym in symbols:
        df = fetch_stock_data(sym, start_date, end_date)
        if not df.empty:
            all_data.append(df)
        else:
            logger.warning(f"⚠️ {sym} no devolvió datos válidos.")

    if not all_data:
        logger.warning("⚠️ Ningún activo fue descargado correctamente.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Datos combinados correctamente ({len(combined)} filas).")
    return combined


# -------------------------------------------------------------------------
# Database Persistence
# -------------------------------------------------------------------------
def store_to_duckdb(df: pd.DataFrame, table_name: str = "prices") -> None:
    """
    Store a DataFrame into a DuckDB database table in append mode.

    Args:
        df (pd.DataFrame): DataFrame to save
        table_name (str): Target table name (default: "prices")
    """
    if df.empty:
        logger.warning("⚠️ No hay datos para guardar en DuckDB.")
        return

    try:
        with duckdb.connect(DATABASE_PATH) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0;"
            )
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df;")
        logger.info(f"✅ Datos guardados en {table_name} ({len(df)} filas).")

    except Exception as e:
        logger.error(f"❌ Error al guardar datos en DuckDB: {e}")
