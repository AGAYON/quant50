import os
from datetime import datetime, timedelta
from importlib import reload

import pandas as pd
import pytest


@pytest.fixture()
def tmp_duckdb_path(tmp_path, monkeypatch):
    db_path = tmp_path / "market.duckdb"
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    # Recargar config y módulo data para que tomen el nuevo DUCKDB_PATH
    import app.utils.config as config
    reload(config)
    from app.services import data as data_module  # type: ignore
    reload(data_module)
    return str(db_path)


def test_upsert_and_latest_timestamp(tmp_duckdb_path):
    # Importar después de fijar DUCKDB_PATH
    from app.services import data as data_module

    # Construir un DataFrame mínimo con 3 días
    base_ts = pd.to_datetime("2024-10-01")
    df = pd.DataFrame(
        {
            "timestamp": [base_ts + pd.Timedelta(days=i) for i in range(3)],
            "symbol": ["AAPL"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )

    data_module.upsert_ohlcv(df, window_days=180)
    latest = data_module.get_latest_timestamp("AAPL")
    assert latest is not None
    assert latest.normalize() == (base_ts + pd.Timedelta(days=2)).normalize()

    # Insertar un día adicional y verificar actualización
    df2 = pd.DataFrame(
        {
            "timestamp": [base_ts + pd.Timedelta(days=3)],
            "symbol": ["AAPL"],
            "open": [103.0],
            "high": [104.0],
            "low": [102.0],
            "close": [103.5],
            "volume": [1300],
        }
    )
    data_module.upsert_ohlcv(df2, window_days=180)
    latest2 = data_module.get_latest_timestamp("AAPL")
    assert latest2.normalize() == (base_ts + pd.Timedelta(days=3)).normalize()


def test_sync_symbol_downloads_only_missing_days(monkeypatch, tmp_duckdb_path):
    # Importar después de fijar DUCKDB_PATH
    from app.services import data as data_module

    # Simular estado inicial: ya hay datos hasta 2024-10-02
    base_ts = pd.to_datetime("2024-10-01")
    df_initial = pd.DataFrame(
        {
            "timestamp": [base_ts + pd.Timedelta(days=i) for i in range(2)],
            "symbol": ["MSFT"] * 2,
            "open": [300.0, 301.0],
            "high": [301.0, 302.0],
            "low": [299.0, 300.0],
            "close": [300.5, 301.5],
            "volume": [2000, 2100],
        }
    )
    data_module.upsert_ohlcv(df_initial, window_days=180)

    # Mockear fetch_stock_data_alpaca para devolver solo días faltantes
    def _mock_fetch(symbol, start_date, end_date):
        # start_date debería comenzar en 2024-10-03
        assert symbol == "MSFT"
        assert start_date >= "2024-10-03"
        return pd.DataFrame(
            {
                "timestamp": [pd.to_datetime("2024-10-03")],
                "symbol": ["MSFT"],
                "open": [302.0],
                "high": [303.0],
                "low": [301.0],
                "close": [302.5],
                "volume": [2200],
            }
        )

    monkeypatch.setattr(data_module, "fetch_stock_data_alpaca", _mock_fetch)

    data_module.sync_symbol("MSFT", window_days=180)
    latest = data_module.get_latest_timestamp("MSFT")
    assert latest.normalize() == pd.to_datetime("2024-10-03").normalize()


