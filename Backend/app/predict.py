import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost.pkl"
ENCODER_PATH = BASE_DIR / "model" / "target_encoder.pkl"

model = joblib.load(MODEL_PATH)
target_encoder = joblib.load(ENCODER_PATH)


def predict_personality(data):
    """
    Predict Introvert or Extrovert from input data.
    """

    # Convert Yes/No into numeric values
    stage_fear = 1 if data.Stage_fear.lower() == "yes" else 0
    drained = 1 if data.Drained_after_socializing.lower() == "yes" else 0

    input_df = pd.DataFrame([{
        "Time_spent_Alone": data.Time_spent_Alone,
        "Stage_fear": stage_fear,
        "Social_event_attendance": data.Social_event_attendance,
        "Going_outside": data.Going_outside,
        "Drained_after_socializing": drained,
        "Friends_circle_size": data.Friends_circle_size,
        "Post_frequency": data.Post_frequency,
        "Social_Activity_Score": data.Social_Activity_Score,
        "Isolation_Index": data.Isolation_Index
    }])

    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    personality = target_encoder.inverse_transform(prediction)[0]

    confidence = float(max(probabilities[0]))

    return {
        "prediction": personality,
        "confidence": round(confidence * 100, 2)
    }