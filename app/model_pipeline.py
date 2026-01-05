# app/model_pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
import joblib
from .preprocessing import preprocess_raw_data
from .config import settings


class NetSpaPipeline:
    """
    Pipeline entraînement + prédiction pour NET_SPA
    with stacking binary (TAIL_PRED) +  regression CatBoost.
    """

    def __init__(
        self,
        cat_features: List[str],
        cls_params: Dict[str, Any] | None = None,
        reg_params: Dict[str, Any] | None = None,
    ):
        self.cat_features = cat_features  # noms de colonnes catégorielles
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

    # ---------- Entraînement ----------

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
        self.features = list(feature_cols)  # on fige l'ordre

        # indices CatBoost
        cat_idx = [self.features.index(c) for c in self.cat_features if c in self.features]

        # 1) Classifieur binaire
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

        # 2)
        df_aug = df.copy()
        df_aug["TAIL_PRED"] = tail_pred
        X_reg = df_aug[self.features]  # maintenant features = features_aug

        cat_idx_aug = [i for i, col in enumerate(self.features) if col in self.cat_features]
        pool_reg = Pool(X_reg, cat_features=cat_idx_aug)

        y_scaled_pred = self.reg_model.predict(pool_reg)
        y_pred = self.y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()
        return y_pred

    # ---------- Sauvegarde / chargement ----------

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
# app/model_pipeline.py (à la fin)
from .config import settings

def train_model(features_path: str | None = None):
    from pathlib import Path

    if features_path is None:
        features_path = Path(settings.DATA_DIR) / "features.parquet"

    df = pd.read_parquet(features_path)

    # ---- définir tes colonnes ici ----
    target = "NET_SPA"

    feature_cols = [c for c in df.columns if c != target]

    # binaire (exemple) : tu peux adapter
    y_reg = df[target]
    y_bin = (df[target] > df[target].median()).astype(int)

    cat_features = [
        c for c in feature_cols
        if df[c].dtype == "object"
    ]

    pipeline = NetSpaPipeline(cat_features=cat_features)
    pipeline.fit(df, feature_cols, y_reg, y_bin)

    pipeline.save(settings.MODELS_DIR)

    return {
        "n_samples": len(df),
        "n_features": len(feature_cols),
    }


def run_preprocessing() -> str:
    raw_path = Path(settings.DATA_DIR) / "SiriusOSS_export.xlsx"
    processed_path = Path(settings.DATA_DIR) / "features.parquet"

    df = preprocess_raw_data(raw_path)  # ici tu fais TOUT ce que tu m'as envoyé

    df.to_parquet(processed_path, index=False)
    return str(processed_path)