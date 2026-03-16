from pathlib import Path
from .config import settings
from .s3_store import download_to_path, s3_key

ARTIFACT_FILES = [
    "title_top_ngrams.pkl",
    "w2v_title.model",
    "w2v_pca.pkl",
    "sbert_pca.pkl",
    "metrics.json",
]

def sync_artifacts_from_s3() -> None:
    models_dir = Path(settings.MODELS_DIR)
    for filename in ARTIFACT_FILES:
        key = s3_key("models", filename)
        download_to_path(key, models_dir / filename)