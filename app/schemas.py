# app/schemas.py
from pydantic import BaseModel

class PredictRequest(BaseModel):
    """
    to be defined
    """



class PredictResponse(BaseModel):
    net_spa: float
