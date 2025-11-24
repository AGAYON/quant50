import os
from datetime import datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
import requests

from app.utils.config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_DATA_URL,
    DUCKDB_PATH,
)


def fetch_stock_data_alpaca(
    symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch daily OHLCV price data for a given symbol from Alpaca Market Data API v2.
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
        "limit": 1000,
        "feed": "iex",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        # Manejo de errores HTTP
        if response.status_code == 401:
            print("❌ Unauthorized (401): revisa tus claves Alpaca.")
            return pd.DataFrame()
        elif response.status_code == 429:
            print("⚠️ Too Many Requests (429): espera unos segundos y reintenta.")
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


def get_latest_timestamp(symbol: str) -> Optional[pd.Timestamp]:
    """
    Return the most recent timestamp we have stored for a given symbol
    in the local DuckDB database.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "AAPL".

    Returns
    -------
    pd.Timestamp or None
        The latest timestamp we have for that symbol in bars_daily.
        Returns None if the table doesn't exist or the symbol has no data.
    """
    # Si la DB no existe todavía, no hay datos.
    if not os.path.exists(DUCKDB_PATH):
        return None

    con = duckdb.connect(DUCKDB_PATH)
    try:
        # Verificar si la tabla bars_daily existe
        table_exists = con.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_name = 'bars_daily';
            """
        ).fetchone()[0]

        if table_exists == 0:
            # No hemos creado la tabla aún
            return None

        result = con.execute(
            """
            SELECT max(timestamp)
            FROM bars_daily
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()

        # result será una tupla tipo (Timestamp or None,)
        latest_ts = result[0]

        if latest_ts is None:
            return None

        # Normalizamos a pandas.Timestamp (timezone-aware o naive, ok)
        return pd.to_datetime(latest_ts)

    finally:
        con.close()


def upsert_ohlcv(df: pd.DataFrame, window_days: int = 180) -> None:
    """
    Inserta o actualiza datos OHLCV en DuckDB, manteniendo una ventana móvil.

    Esta versión recrea la tabla 'bars_daily' en cada actualización.
    Es eficiente para bases pequeñas y evita conflictos de clave.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas ['timestamp', 'symbol',
        'open', 'high', 'low', 'close', 'volume'].
    window_days : int, optional
        Tamaño de la ventana móvil en días (default = 180).

    Returns
    -------
    None
    """
    if df.empty:
        print("⚠️ No hay datos nuevos para insertar.")
        return

    con = duckdb.connect(DUCKDB_PATH)

    # 1️⃣ Crear tabla si no existe (estructura vacía)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bars_daily AS
        SELECT * FROM df LIMIT 0
        """
    )

    # 2️⃣ Leer datos existentes y concatenar con nuevos
    existing = con.execute("SELECT * FROM bars_daily").fetchdf()
    combined = pd.concat([existing, df], ignore_index=True)

    # 3️⃣ Eliminar duplicados por (timestamp, symbol)
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

    # 4️⃣ Aplicar ventana móvil (solo últimos N días)
    latest_date = combined["timestamp"].max()
    cutoff_date = latest_date - timedelta(days=window_days)
    combined = combined[combined["timestamp"] >= cutoff_date]

    # 5️⃣ Reemplazar la tabla por la versión actualizada
    con.execute("DROP TABLE bars_daily")
    con.execute("CREATE TABLE bars_daily AS SELECT * FROM combined")

    con.close()
    print(
        f"✅ Datos actualizados hasta {latest_date.date()} (ventana {window_days} días)."
    )


def sync_symbol(symbol: str, window_days: int = 180) -> None:
    """
    Sincroniza los datos más recientes de un símbolo específico desde Alpaca.

    Si no hay datos en la base local, descarga el histórico completo permitido.
    Si ya existen, descarga solo los días faltantes.

    Parameters
    ----------
    symbol : str
        Ticker del activo (ej. 'AAPL').
    window_days : int, optional
        Ventana de días a mantener en la base (default = 180).

    Returns
    -------
    None
    """
    from app.services.data import (
        fetch_stock_data_alpaca,
        get_latest_timestamp,
        upsert_ohlcv,
    )

    latest_ts = get_latest_timestamp(symbol)
    today = datetime.utcnow().date()

    if latest_ts is None:
        # No hay datos previos → descargar histórico completo (por ejemplo, 180 días)
        start_date = today - timedelta(days=window_days)
        print(f"⬇️ {symbol}: descargando histórico inicial desde {start_date}...")
    else:
        # Solo descargar días faltantes
        start_date = latest_ts.date() + timedelta(days=1)
        print(f"🔄 {symbol}: actualizando desde {start_date} hasta {today}...")

    # Evitar descargas redundantes
    if start_date > today:
        print(f"✅ {symbol}: ya está actualizado.")
        return

    df_new = fetch_stock_data_alpaca(symbol, str(start_date), str(today))

    if df_new.empty:
        print(f"⚠️ {symbol}: no se recibieron datos nuevos.")
        return

    upsert_ohlcv(df_new, window_days=window_days)
    print(
        f"✅ {symbol}: sincronización completa hasta {df_new['timestamp'].max().date()}."
    )


def get_all_bars() -> pd.DataFrame:
    """
    Retrieve all daily bars from DuckDB.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, symbol, open, high, low, close, volume
    """
    if not os.path.exists(DUCKDB_PATH):
        return pd.DataFrame()

    con = duckdb.connect(DUCKDB_PATH)
    try:
        # Check if table exists
        table_exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'bars_daily'"
        ).fetchone()[0]

        if table_exists == 0:
            return pd.DataFrame()

        df = con.execute("SELECT * FROM bars_daily").fetchdf()
        return df
    finally:
        con.close()
