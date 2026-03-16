from fastapi import FastAPI
from .api import router as api_router
from .artifacts import sync_artifacts_from_s3

app = FastAPI(
    title="PageCount Forecasting API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
@app.on_event("startup")
def startup():
    sync_artifacts_from_s3()

app.include_router(api_router, prefix="/api")