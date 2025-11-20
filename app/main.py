from fastapi import FastAPI

from app.routes import router
from app.utils import config

app = FastAPI(title="Quant50 Portfolio API", version="0.1")

app.include_router(router)


@app.get("/")
def root():
    """
    Endpoint base para probar conexión con variables de entorno.
    """
    return {
        "message": "Quant50 API running 🚀",
        "alpaca_base_url": config.ALPACA_BASE_URL,
        "reports_dir": config.REPORTS_DIR,
    }
