# app/model_pipeline.py
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier, CatBoostRegressor

from .config import settings
from .preprocessing import preprocess_one_record
from .s3_utils import download_file_from_s3, upload_file_to_s3


MODELS_DIR = Path(settings.MODELS_DIR)
DATA_DIR = Path(settings.DATA_DIR)




def sigmoid_rmse_score(rmse: float, std_y: float, k: float = 5.0, t: float = 0.8) -> float:
    if std_y == 0 or np.isnan(std_y):
        return float("nan")
    normalized_rmse = rmse / std_y
    return float(1 / (1 + np.exp(k * (normalized_rmse - t))))





def _load_predict_artifacts():
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    sync_predict_dependencies_from_s3()

    with open(models_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols") or meta.get("features") or meta.get("columns")
    if feature_cols is None:
        raise RuntimeError("meta.json is empty!")

    reg_model = CatBoostRegressor()
    reg_model.load_model(str(models_dir / "reg_model.cbm"))

    cls_model = CatBoostClassifier()
    cls_model.load_model(str(models_dir / "cls_model.cbm"))

    scaler = joblib.load(models_dir / "scaler.pkl")

    return feature_cols, cls_model, reg_model, scaler

def sync_predict_dependencies_from_s3():
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "meta.json",
        "reg_model.cbm",
        "cls_model.cbm",
        "scaler.pkl",
        "title_top_ngrams.pkl",
        "w2v_title.model",
        "w2v_pca.pkl",
        "sbert_pca.pkl",
        "title_counts.pkl",
    ]

    for filename in required_files:
        download_file_from_s3(
            f"{settings.S3_MODELS_PREFIX}/{filename}",
            models_dir / filename
        )

def predict(payload):
    feature_cols, cls_model, reg_model, scaler = _load_predict_artifacts()

    raw = pd.DataFrame([payload.model_dump()])
    feats = preprocess_one_record(raw)

    base_cols = feature_cols[:]
    X_cls = feats.reindex(columns=base_cols, fill_value=0)

    tail_pred = cls_model.predict(X_cls)
    tail_pred = pd.Series(tail_pred).astype(int).to_numpy().flatten()
    feats["TAIL_PRED"] = int(tail_pred[0])

    if "TAIL_PRED" in base_cols:
        features_aug = base_cols
    else:
        features_aug = base_cols + ["TAIL_PRED"]

    X_reg = feats.reindex(columns=features_aug, fill_value=0)

    y_scaled = reg_model.predict(X_reg)
    y_scaled = float(y_scaled[0])

    y_real = scaler.inverse_transform([[y_scaled]])[0][0]
    net_spa = float(y_real)

    return net_spa, feats.iloc[0].to_dict(), X_reg.iloc[0].to_dict()
