# app/model_pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .config import settings
from .preprocessing import preprocess_raw_data, preprocess_one_record
from .s3_utils import download_file_from_s3, upload_file_to_s3


MODELS_DIR = Path(settings.MODELS_DIR)
DATA_DIR = Path(settings.DATA_DIR)


class NetSpaPipeline:
    """
    Pipeline training + prediction for NET_SPA
    with stacking binary (TAIL_PRED) + regression CatBoost.
    """

    def __init__(
        self,
        cat_features: List[str],
        cls_params: Dict[str, Any] | None = None,
        reg_params: Dict[str, Any] | None = None,
    ):
        self.cat_features = cat_features
        self.cls_params = cls_params or dict(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
        self.reg_params = reg_params or dict(
            iterations=500,
            learning_rate=0.1,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )

        self.cls_model: CatBoostClassifier | None = None
        self.reg_model: CatBoostRegressor | None = None
        self.y_scaler: StandardScaler | None = None
        self.features: List[str] | None = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        y_reg: pd.Series,
        y_bin: pd.Series,
    ) -> "NetSpaPipeline":
        self.features = list(feature_cols)

        cat_idx = [self.features.index(c) for c in self.cat_features if c in self.features]

        # 1) binary classifier
        X_cls = df[self.features]
        train_pool_cls = Pool(X_cls, y_bin, cat_features=cat_idx)

        self.cls_model = CatBoostClassifier(**self.cls_params)
        self.cls_model.fit(train_pool_cls)

        # 2) stacking
        tail_pred = self.cls_model.predict(X_cls).astype(int).flatten()
        df_aug = df.copy()
        df_aug["TAIL_PRED"] = tail_pred

        features_aug = self.features + ["TAIL_PRED"]
        X_reg = df_aug[features_aug]

        cat_idx_aug = [i for i, col in enumerate(features_aug) if col in self.cat_features]

        # 3) scale target
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y_reg.to_numpy().reshape(-1, 1)).flatten()

        train_pool_reg = Pool(X_reg, y_scaled, cat_features=cat_idx_aug)

        self.reg_model = CatBoostRegressor(**self.reg_params)
        self.reg_model.fit(train_pool_reg)

        self.features = features_aug
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.cls_model is None or self.reg_model is None or self.y_scaler is None:
            raise RuntimeError("Pipeline not fitted or not loaded correctly.")

        base_features = [c for c in self.features if c != "TAIL_PRED"]
        X_cls = df[base_features]
        cat_idx = [base_features.index(c) for c in self.cat_features if c in base_features]

        pool_cls = Pool(X_cls, cat_features=cat_idx)
        tail_pred = self.cls_model.predict(pool_cls).astype(int).flatten()

        df_aug = df.copy()
        df_aug["TAIL_PRED"] = tail_pred
        X_reg = df_aug[self.features]

        cat_idx_aug = [i for i, col in enumerate(self.features) if col in self.cat_features]
        pool_reg = Pool(X_reg, cat_features=cat_idx_aug)

        y_scaled_pred = self.reg_model.predict(pool_reg)
        y_pred = self.y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()
        return y_pred

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        assert self.cls_model is not None
        assert self.reg_model is not None
        assert self.y_scaler is not None
        assert self.features is not None

        self.cls_model.save_model(folder / "cls_model.cbm")
        self.reg_model.save_model(folder / "reg_model.cbm")
        joblib.dump(self.y_scaler, folder / "scaler.pkl")

        meta = {
            "features": self.features,
            "cat_features": self.cat_features,
            "cls_params": self.cls_params,
            "reg_params": self.reg_params,
        }
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "NetSpaPipeline":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))

        pipeline = cls(
            cat_features=meta["cat_features"],
            cls_params=meta["cls_params"],
            reg_params=meta["reg_params"],
        )
        pipeline.features = meta["features"]

        pipeline.cls_model = CatBoostClassifier()
        pipeline.cls_model.load_model(str(folder / "cls_model.cbm"))

        pipeline.reg_model = CatBoostRegressor()
        pipeline.reg_model.load_model(str(folder / "reg_model.cbm"))

        pipeline.y_scaler = joblib.load(folder / "scaler.pkl")
        return pipeline


def sigmoid_rmse_score(rmse: float, std_y: float, k: float = 5.0, t: float = 0.8) -> float:
    if std_y == 0 or np.isnan(std_y):
        return float("nan")
    normalized_rmse = rmse / std_y
    return float(1 / (1 + np.exp(k * (normalized_rmse - t))))


def run_preprocessing() -> str:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "SiriusOSS_export.xlsx"

    download_file_from_s3(settings.S3_RAW_KEY, raw_path)

    df = preprocess_raw_data(raw_path)

    processed_path = DATA_DIR / "features.parquet"
    df.to_parquet(processed_path, index=False)

    return str(processed_path)


def train_model(
    features_path: str | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
):

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if features_path is None:
        local_features_path = DATA_DIR / "features.parquet"
        download_file_from_s3(settings.S3_FEATURES_KEY, local_features_path)
        features_path = local_features_path
    else:
        features_path = Path(features_path)

    df = pd.read_parquet(features_path).replace({None: np.nan})

    feature_cols = [
        "AM_GROUPING", "DOSSIER_TYPE",
        "PROC_DOC_COMBO", "Committee_regrouped",
        "DOC_DOCEP_COMBO", "DOC_TYPE_PROCNATURE",
        "Month", "DayOfWeek", "IsWeekend", "Quarter",
        "ROLE", "DOC_EP_TEMPLATE",
        "Procedure_Family", "PROC_DOC_TYPE",
        "DOC_TYPE", "PROC_TYPE_NATURE",
        "PROC_NATURE", "TITLE_FREQ",
        "TITLE_WORD_COUNT", "TITLE_CHAR_COUNT"
    ]

    word2vec_features = [col for col in df.columns if col.startswith("TITLE_W2V_")]
    sbert_features = [col for col in df.columns if col.startswith("TITLE_SBERT_")]
    ngram_features = [col for col in df.columns if col.startswith("TITLE_ngram_")]

    feature_cols = feature_cols + word2vec_features + sbert_features + ngram_features

    target = "NET_SPA"
    y_reg = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        y_reg,
        test_size=test_size,
        random_state=random_state,
    )

    cutoff = float(y_train.median())
    y_bin_train = (y_train > cutoff).astype(int)

    cat_features = [c for c in feature_cols if X_train[c].dtype == "object"]

    pipeline = NetSpaPipeline(cat_features=cat_features)
    pipeline.fit(
        df=pd.concat([X_train, y_train.rename(target)], axis=1),
        feature_cols=feature_cols,
        y_reg=y_train,
        y_bin=y_bin_train,
    )

    y_pred_test = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    std_y = float(np.std(y_test))
    sigmoid_score = sigmoid_rmse_score(rmse, std_y)

    # save local model artifacts
    pipeline.save(MODELS_DIR)

    df_test = df.loc[X_test.index].copy()
    df_test["NEW_PREDICTION"] = y_pred_test.round(2)
    df_test_path = MODELS_DIR / "dfTest.xlsx"
    df_test.to_excel(df_test_path, index=False)

    metrics = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "cutoff_bin_median_train": cutoff,
        "rmse_test": rmse,
        "r2_test": r2,
        "std_y_test": std_y,
        "sigmoid_rmse_test": float(sigmoid_score),
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # upload training outputs to S3
    upload_file_to_s3(MODELS_DIR / "cls_model.cbm", f"{settings.S3_MODELS_PREFIX}/cls_model.cbm")
    upload_file_to_s3(MODELS_DIR / "reg_model.cbm", f"{settings.S3_MODELS_PREFIX}/reg_model.cbm")
    upload_file_to_s3(MODELS_DIR / "scaler.pkl", f"{settings.S3_MODELS_PREFIX}/scaler.pkl")
    upload_file_to_s3(MODELS_DIR / "meta.json", f"{settings.S3_MODELS_PREFIX}/meta.json")
    upload_file_to_s3(metrics_path, f"{settings.S3_MODELS_PREFIX}/metrics.json")
    upload_file_to_s3(df_test_path, f"{settings.S3_MODELS_PREFIX}/dfTest.xlsx")

    return metrics


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