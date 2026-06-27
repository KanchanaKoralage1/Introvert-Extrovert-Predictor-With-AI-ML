from fastapi import FastAPI

from app.schemas import PersonalityRequest
from app.predict import predict_personality

app = FastAPI(
    title="Introvert vs Extrovert Prediction API",
    description="Machine Learning API built with FastAPI and XGBoost",
    version="1.0.0",
)


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