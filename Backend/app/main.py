from fastapi import FastAPI
from app.predict import router

from app.schemas import PersonalityRequest
from app.predict import predict_personality

app = FastAPI(
    title="Introvert vs Extrovert Prediction API",
    description="Machine Learning API built with FastAPI and XGBoost",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

@app.get("/")
def home():
    return {
        "message": "Introvert-Extrovert Prediction API is running."
    }


@app.post("/predict")
def predict(request: PersonalityRequest):

    result = predict_personality(request)

    return result

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost"
    }