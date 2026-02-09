# app/schemas.py
from typing import Dict, Any, Optional
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    TITLE: str
    CREATED_1: str
    DOC_TYPE: str
    DOSSIER_TYPE: str
    PROC_TYPE: str
    PROC_NATURE: str
    ROLE: str
    DOC_EP_TEMPLATE: str
    COMMITTEE_1: str

class PredictResponse(BaseModel):
    net_spa: float
    features: Dict[str, Any]
    aligned_features: Optional[Dict[str, Any]] = None



class PredictRequest(BaseModel):
    TITLE: str
    CREATED_1: Optional[datetime] = None
    DOC_TYPE: Optional[str] = None
    DOSSIER_TYPE: Optional[str] = None
    PROC_TYPE: Optional[str] = None
    PROC_NATURE: Optional[str] = None
    ROLE: Optional[str] = None
    DOC_EP_TEMPLATE: Optional[str] = None
    COMMITTEE_1: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_none(cls, v: Any):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("CREATED_1", mode="before")
    @classmethod
    def parse_created(cls, v):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if isinstance(v, datetime):
            return v
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                pass
        raise ValueError("CREATED_1: Date format is invalid (ex: 2021-06-11 18:53:18.799)")
