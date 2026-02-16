# app/api.py
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import json

from . import model_pipeline
from .schemas import PredictRequest, PredictResponsePublic, PredictResponseDebug
from .config import settings
from .jobs import submit_job, get_job

router = APIRouter(prefix="/api")

# ---------- ASYNC JOBS (submit) ----------

@router.post("/extract")
def extract():
    job = submit_job("extract", model_pipeline.run_extraction)
    return {"status": "accepted", "job_id": job.id}

@router.post("/preprocess")
def preprocess():
    job = submit_job("preprocess", model_pipeline.run_preprocessing)
    return {"status": "accepted", "job_id": job.id}

@router.post("/train")
def train():
    job = submit_job("train", model_pipeline.train_model)
    return {"status": "accepted", "job_id": job.id}

@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

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
    return {"health": "green"}
