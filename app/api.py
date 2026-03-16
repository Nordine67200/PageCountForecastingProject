# app/api.py
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import json

from . import model_pipeline
from .schemas import PredictRequest, PredictResponsePublic, PredictResponseDebug
from .config import settings
from .jobs import submit_job, get_job

router = APIRouter()



# ---------- PREDICT (sync) ----------

@router.post("/predict")
def predict(payload: PredictRequest, debug: bool = Query(False)):
    net_spa, features, aligned_features = model_pipeline.predict(payload)

    if debug:
        return PredictResponseDebug(
            net_spa=net_spa,
            features=features,
            aligned_features=aligned_features,
        )

    return PredictResponsePublic(net_spa=net_spa)
# ---------- METRICS / HEALTH ----------

@router.get("/getMetrics")
def getMetrics():
    metrics_path = Path(settings.MODELS_DIR) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Run /train first.")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/health")
def health():
    models_dir = Path(settings.MODELS_DIR)

    required_files = [
        "cls_model.cbm",
        "reg_model.cbm",
        "scaler.pkl",
        "meta.json",
    ]

    missing = [f for f in required_files if not (models_dir / f).exists()]

    if missing:
        return {
            "health": "degraded",
            "missing_models": missing
        }

    return {"health": "green"}
