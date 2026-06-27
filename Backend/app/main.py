from fastapi import FastAPI

from app.schemas import PersonalityRequest
from app.predict import predict_personality

app = FastAPI(
    title="Introvert-Extrovert Prediction API",
    description="Predict whether a person is an Introvert or Extrovert using a trained XGBoost model.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Introvert-Extrovert Prediction API is running."
    }


@app.post("/predict")
def predict(request: PersonalityRequest):

    prediction = predict_personality(request)

    return {
        "prediction": prediction
    }