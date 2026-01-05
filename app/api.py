# app/api.py
from fastapi import APIRouter
from . import model_pipeline
from .schemas import PredictRequest, PredictResponse

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