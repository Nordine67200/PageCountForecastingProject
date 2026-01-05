from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # dossier principal pour les données
    DATA_DIR = BASE_DIR / "data"

    # dossier pour sauvegarder modèles + objets NLP
    MODELS_DIR = BASE_DIR / "models"

    # tu pourras ajouter plus tard :
    # DB_URL, API keys, paramètres, etc.

settings = Settings()
