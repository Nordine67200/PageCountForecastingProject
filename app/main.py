from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(
    title="PageCount Forecasting API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.include_router(api_router, prefix="/api")