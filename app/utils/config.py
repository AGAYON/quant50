import os

from dotenv import load_dotenv

# Cargar el archivo .env desde la raíz del proyecto
load_dotenv()

# Variables principales de Alpaca API
ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
)

ALPACA_DATA_URL = os.getenv(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets/v2",
)

# Directorio de reportes
REPORTS_DIR = os.getenv("REPORTS_DIR", "./data/reports")

# Nivel de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
