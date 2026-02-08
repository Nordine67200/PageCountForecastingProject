# app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

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
