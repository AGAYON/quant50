import requests
import pandas as pd
from app.utils.config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_DATA_URL,
)


def fetch_stock_data_alpaca(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV price data for a given symbol from Alpaca Market Data API v2.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g. 'AAPL', 'MSFT').
    start_date : str
        Start date in ISO format 'YYYY-MM-DD'.
    end_date : str
        End date in ISO format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        Returns empty DataFrame if an error or empty response occurs.
    """

    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
    }

    params = {
        "timeframe": "1Day",
        "start": f"{start_date}T00:00:00Z",
        "end": f"{end_date}T23:59:59Z",
        "limit": 1000,  # máximo permitido
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        # Manejo de errores HTTP
        if response.status_code == 401:
            print(f"❌ Unauthorized (401): revisa tus claves Alpaca.")
            return pd.DataFrame()
        elif response.status_code == 429:
            print(f"⚠️ Too Many Requests (429): espera unos segundos y reintenta.")
            return pd.DataFrame()
        elif response.status_code >= 500:
            print(f"⚠️ Server error {response.status_code}: {response.text}")
            return pd.DataFrame()
        elif response.status_code != 200:
            print(f"⚠️ Error HTTP {response.status_code}: {response.text}")
            return pd.DataFrame()

        data = response.json().get("bars", [])
        if not data:
            print(f"⚠️ No data returned for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["symbol"] = symbol

        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request error for {symbol}: {e}")
        return pd.DataFrame()
