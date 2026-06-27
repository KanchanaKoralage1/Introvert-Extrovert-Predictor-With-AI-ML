import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost.pkl"
ENCODER_PATH = BASE_DIR / "model" / "target_encoder.pkl"

model = joblib.load(MODEL_PATH)
target_encoder = joblib.load(ENCODER_PATH)