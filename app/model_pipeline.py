# app/model_pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
import joblib
from .preprocessing import preprocess_raw_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .config import settings


class NetSpaPipeline:
    """
    Pipeline etraining + prediction for NET_SPA
    with stacking binary (TAIL_PRED) +  regression CatBoost.
    """

    def __init__(
        self,
        cat_features: List[str],
        cls_params: Dict[str, Any] | None = None,
        reg_params: Dict[str, Any] | None = None,
    ):
        self.cat_features = cat_features  # categorical columns names
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
        self.features: List[str] | None = None  # features for inference

    # ---------- Training ----------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        y_reg: pd.Series,
        y_bin: pd.Series,
    ) -> "NetSpaPipeline":
        """
        df: dataframe complet d'entraînement
        feature_cols: colonnes X de départ
        y_reg: série NET_SPA brute
        y_bin: série binaire pour le classifieur (0/1)
        """
        self.features = list(feature_cols)

        # Catboost indexes
        cat_idx = [self.features.index(c) for c in self.cat_features if c in self.features]

        # 1) binary classifier
        X_cls = df[self.features]
        train_pool_cls = Pool(X_cls, y_bin, cat_features=cat_idx)

        self.cls_model = CatBoostClassifier(**self.cls_params)
        self.cls_model.fit(train_pool_cls)

        # 2) Prédiction binaire sur le même X (stacking interne)
        tail_pred = self.cls_model.predict(X_cls).astype(int).flatten()
        df_aug = df.copy()
        df_aug["TAIL_PRED"] = tail_pred

        features_aug = self.features + ["TAIL_PRED"]
        X_reg = df_aug[features_aug]


        cat_idx_aug = [i for i, col in enumerate(features_aug) if col in self.cat_features]

        # 3) Standardize y: (y- mean)/std
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y_reg.to_numpy().reshape(-1, 1)).flatten()

        train_pool_reg = Pool(X_reg, y_scaled, cat_features=cat_idx_aug)

        self.reg_model = CatBoostRegressor(**self.reg_params)
        self.reg_model.fit(train_pool_reg)

        # we keep features_aug for prediction
        self.features = features_aug
        return self

    # ---------- Prédiction ----------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        df : dataframe avec au minimum les colonnes nécessaires (self.features sans TAIL_PRED),
        c.-à-d. les features d'origine.
        Retourne NET_SPA (déstandardisé).
        """
        if self.cls_model is None or self.reg_model is None or self.y_scaler is None:
            raise RuntimeError("Pipeline not fitted or not loaded correctly.")

        # 1) binary prediction
        base_features = [c for c in self.features if c != "TAIL_PRED"]
        X_cls = df[base_features]
        cat_idx = [base_features.index(c) for c in self.cat_features if c in base_features]

        pool_cls = Pool(X_cls, cat_features=cat_idx)
        tail_pred = self.cls_model.predict(pool_cls).astype(int).flatten()

        # 2) regression
        df_aug = df.copy()
        df_aug["TAIL_PRED"] = tail_pred
        X_reg = df_aug[self.features]  # features = features_aug

        cat_idx_aug = [i for i, col in enumerate(self.features) if col in self.cat_features]
        pool_reg = Pool(X_reg, cat_features=cat_idx_aug)

        y_scaled_pred = self.reg_model.predict(pool_reg)
        y_pred = self.y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()
        return y_pred

    # ---------- Save/load ----------

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
        (folder / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, folder: str | Path) -> "NetSpaPipeline":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())

        pipeline = cls(
            cat_features=meta["cat_features"],
            cls_params=meta["cls_params"],
            reg_params=meta["reg_params"],
        )
        pipeline.features = meta["features"]

        pipeline.cls_model = CatBoostClassifier()
        pipeline.cls_model.load_model(folder / "cls_model.cbm")

        pipeline.reg_model = CatBoostRegressor()
        pipeline.reg_model.load_model(folder / "reg_model.cbm")

        pipeline.y_scaler = joblib.load(folder / "scaler.pkl")
        return pipeline

from .config import settings

def train_model(
    features_path: str | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
):
    from pathlib import Path

    if features_path is None:
        features_path = Path(settings.DATA_DIR) / "features.parquet"

    df = pd.read_parquet(features_path).replace({None: np.nan})

    feature_cols = [
        'AM_GROUPING', 'DOSSIER_TYPE',
        'PROC_DOC_COMBO', 'Committee_regrouped',
        'DOC_DOCEP_COMBO', 'DOC_TYPE_PROCNATURE',
        'Month', 'DayOfWeek', 'IsWeekend', 'Quarter',
        'ROLE', 'DOC_EP_TEMPLATE',
        'Procedure_Family', 'PROC_DOC_TYPE',
        'DOC_TYPE', 'PROC_TYPE_NATURE',
        'PROC_NATURE', 'TITLE_FREQ',
        'TITLE_WORD_COUNT', 'TITLE_CHAR_COUNT'
    ]

    word2vec_features = [col for col in df.columns if col.startswith('TITLE_W2V_')]
    sbert_features = [col for col in df.columns if col.startswith('TITLE_SBERT_')]
    ngram_features = [col for col in df.columns if col.startswith('TITLE_ngram_')]

    feature_cols = feature_cols + word2vec_features + sbert_features + ngram_features

    target = "NET_SPA"

    # --- y ---
    y_reg = df[target]

    # binaire (exemple) : médiane sur le TRAIN uniquement pour éviter fuite
    # => on va d'abord split puis calculer y_bin sur train et appliquer au train/test via un cutoff.
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        y_reg,
        test_size=test_size,
        random_state=random_state,
    )

    cutoff = float(y_train.median())
    y_bin_train = (y_train > cutoff).astype(int)
    y_bin_test = (y_test > cutoff).astype(int)  # pas strictement nécessaire mais ok

    # --- cat features ---
    # Important: on détecte les cat_features sur le df original (ou X_train)
    cat_features = [c for c in feature_cols if X_train[c].dtype == "object"]

    # --- fit pipeline sur TRAIN uniquement ---
    pipeline = NetSpaPipeline(cat_features=cat_features)
    pipeline.fit(
        df=pd.concat([X_train, y_train.rename(target)], axis=1),
        feature_cols=feature_cols,
        y_reg=y_train,
        y_bin=y_bin_train,
    )

    # --- évaluation sur TEST ---
    y_pred_test = pipeline.predict(X_test)

    # métriques
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    std_y = float(np.std(y_test))
    sigmoid_score = sigmoid_rmse_score(rmse, std_y)

    # --- save modèle ---
    pipeline.save(settings.MODELS_DIR)

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

    # --- save metrics json ---
    metrics_dir = Path(settings.MODELS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- API response ---
    return metrics



def run_preprocessing() -> str:
    raw_path = Path(settings.DATA_DIR) / "SiriusOSS_export.xlsx"
    processed_path = Path(settings.DATA_DIR) / "features.parquet"

    df = preprocess_raw_data(raw_path)  # ici tu fais TOUT ce que tu m'as envoyé

    df.to_parquet(processed_path, index=False)
    return str(processed_path)


def sigmoid_rmse_score(rmse: float, std_y: float, k: float = 5.0, t: float = 0.8) -> float:
    """
    Sigmoid score based on RMSE normalized by standard deviation.
    The smaller the RMSE, the closer the score is to 1.
    """
    if std_y == 0 or np.isnan(std_y):
        return float("nan")
    normalized_rmse = rmse / std_y
    return float(1 / (1 + np.exp(k * (normalized_rmse - t))))
