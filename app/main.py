# app/main.py
from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(title="PageCount Forecasting API")

app.include_router(api_router)




# launch locally with cmd `python app/main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
