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

import yfinance as yf
import pandas as pd
import duckdb
import logging
from datetime import datetime
from app.utils.config import DATABASE_PATH

# Configurar logging
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock prices for an asset using yfinance.
    
    Args:
        symbol (str): Asset ticker (e.g., 'AAPL', 'SPY')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with columns [Date, Open, High, Low, Close,
                     Adj Close, Volume, Symbol]
    """

    import os
    os.environ["YFINANCE_USE_CURL_CFFI"] = "false"

    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            logger.warning(f"⚠️ No se obtuvieron datos para {symbol}.")
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        df["Symbol"] = symbol
        return df
    
    except Exception as e:
        logger.error(f"❌ Error descargando {symbol}: {e}")
        return pd.DataFrame()


def fetch_multiple_symbols(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads multiple assets and concatenates the results.
    
    Args:
        symbols (list[str]): List of asset tickers to download
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Concatenated DataFrame with all symbols data
    """
    all_data = []
    for sym in symbols:
        df = fetch_stock_data(sym, start_date, end_date)
        if not df.empty:
            all_data.append(df)
    if not all_data:
        logger.warning("⚠️ Ningún activo descargado correctamente.")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def store_to_duckdb(df: pd.DataFrame, table_name: str = "prices") -> None:
    """
    Saves a DataFrame to DuckDB database in append mode.
    
    Args:
        df (pd.DataFrame): DataFrame to save to database
        table_name (str): Name of the table to store data (default: "prices")
    """
    if df.empty:
        logger.warning("⚠️ No hay datos para guardar en DuckDB.")
        return
    
    try:
        conn = duckdb.connect(DATABASE_PATH)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0;")
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM df;")
        conn.close()
        logger.info(f"✅ Datos guardados en {table_name} ({len(df)} filas).")
    except Exception as e:
        logger.error(f"❌ Error al guardar datos en DuckDB: {e}")
