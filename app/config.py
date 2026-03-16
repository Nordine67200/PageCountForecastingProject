from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    S3_BUCKET: str = "pagecount-forecasting-bucket"
    AWS_REGION: str = "eu-west-1"

    # S3 keys / prefixes
    S3_ROOT_PREFIX: str = "pagecount"
    S3_RAW_KEY: str = "pagecount/raw/SiriusOSS_export.xlsx"
    S3_FEATURES_KEY: str = "pagecount/processed/features.parquet"

    S3_RAW_PREFIX: str = "pagecount/raw"
    S3_PROCESSED_PREFIX: str = "pagecount/processed"
    S3_MODELS_PREFIX: str = "pagecount/models"
    S3_METRICS_PREFIX: str = "pagecount/metrics"

    # Local working directories
    DATA_DIR: str = "/tmp/app/data"
    MODELS_DIR: str = "/tmp/app/models"


settings = Settings()