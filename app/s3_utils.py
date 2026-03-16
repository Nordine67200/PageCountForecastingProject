# app/s3_utils.py
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

from .config import settings

s3 = boto3.client("s3", region_name=settings.AWS_REGION)


def download_file_from_s3(s3_key: str, local_path: str | Path) -> Path:

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f'path on S3: {settings.S3_BUCKET}, s3_key: {s3_key}, local path: {str(local_path)}')
        s3.download_file(settings.S3_BUCKET, s3_key, str(local_path))
    except ClientError as e:
        raise RuntimeError(f"Failed to download s3://{settings.S3_BUCKET}/{s3_key}") from e

    return local_path


def upload_file_to_s3(local_path: str | Path, s3_key: str) -> None:

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    try:
        s3.upload_file(str(local_path), settings.S3_BUCKET, s3_key)
    except ClientError as e:
        raise RuntimeError(f"Failed to upload {local_path} to s3://{settings.S3_BUCKET}/{s3_key}") from e