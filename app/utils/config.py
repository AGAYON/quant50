# app/utils/config.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Fuente de datos
DATA_SOURCE = os.getenv("DATA_SOURCE", "yfinance")

# Directorios y rutas
REPORTS_DIR = os.getenv("REPORTS_DIR", "./data/reports")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/market.duckdb")

# Nivel de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

