from unittest.mock import patch

import pandas as pd
import pytest

from app.services.data import fetch_stock_data_alpaca


@pytest.fixture
def mock_response_ok():
    """Simula una respuesta JSON de Alpaca API."""
    return {
        "bars": [
            {
                "t": "2024-10-01T00:00:00Z",
                "o": 100.0,
                "h": 105.0,
                "l": 99.0,
                "c": 104.0,
                "v": 5000,
            },
            {
                "t": "2024-10-02T00:00:00Z",
                "o": 104.0,
                "h": 106.0,
                "l": 103.0,
                "c": 105.0,
                "v": 6000,
            },
        ]
    }


@patch("app.services.data.requests.get")
def test_fetch_stock_data_alpaca(mock_get, mock_response_ok):
    """Verifica que fetch_stock_data_alpaca devuelva un DataFrame correcto."""
    # Configurar el mock para simular la API
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response_ok

    df = fetch_stock_data_alpaca("AAPL", "2024-10-01", "2024-10-02")

    # Validar tipo de retorno
    assert isinstance(df, pd.DataFrame)

    # Validar columnas esperadas
    expected_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols

    # Validar contenido simulado
    assert len(df) == 2
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["close"].iloc[1] == 105.0
