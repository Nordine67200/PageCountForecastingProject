# app/main.py
from fastapi import FastAPI
from .api import router as api_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="PageCount Forecasting API")

app.include_router(api_router)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




