# app/api.py
from fastapi import APIRouter
from . import model_pipeline
from .schemas import PredictRequest, PredictResponse
from pathlib import Path
from .config import settings
import json
from fastapi import HTTPException

router = APIRouter()

@router.post("/extract")
def extract():
    # launch extraction
    filepath = model_pipeline.run_extraction()
    return {"status": "ok", "excel_path": filepath}


@router.post("/train")
def train():
    metrics = model_pipeline.train_model()
    return {"status": "ok", "metrics": metrics}


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    net_spa = model_pipeline.predict(payload)
    return PredictResponse(net_spa=net_spa)

@router.post("/preprocess")
def preprocess():
    path = model_pipeline.run_preprocessing()
    return {"status": "ok", "features_path": path}

@router.get("/getMetrics")
def getMetrics():
    metrics_path = Path(settings.MODELS_DIR) / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run /train first."
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/health")
def health():
    return {"health": "green"}