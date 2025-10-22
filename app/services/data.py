"""
app/services/data.py
~~~~~~~~~~~~~~~~~~~~

Module responsible for financial data ingestion from the
Alpaca Market Data v2 API.

Main functions:
- fetch_stock_data: downloads daily OHLCV prices for a given symbol.
- (future) batch_fetch_stocks: multiple symbol download for portfolio
  optimization.

Best practices:
- Reading credentials from app/utils/config.py
- Robust HTTP error and timeout handling
- Returns clean DataFrames ready for analysis or modeling

Author: Andrés Gayón
Project: quant50
"""

import pandas as pd


def fetch_stock_data(symbol: str, 
                     start_date: str, 
                     end_date: str) -> pd.DataFrame:
    """
    Downloads daily OHLCV prices for a symbol using Alpaca Market Data
    API v2.

    Parameters
    ----------
    symbol : str
        Asset ticker (e.g., "AAPL", "SPY").
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close',
        'volume']. If an error occurs, returns an empty DataFrame.
    """
    raise NotImplementedError("fetch_stock_data not yet implemented.")
