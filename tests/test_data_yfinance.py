from app.services.data import fetch_stock_data, store_to_duckdb
import pandas as pd

# Probar descarga
df = fetch_stock_data("AAPL", "2024-10-01", "2024-10-10")
print("✅ Datos descargados:")
print(df.head())

# Guardar en DuckDB
if not df.empty:
    store_to_duckdb(df)
    print("✅ Datos guardados correctamente en DuckDB.")
else:
    print("⚠️ DataFrame vacío, no se guardó nada.")
