from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# =========================
# APP INIT
# =========================
app = FastAPI(title="Customer Churn API", version="1.0")

# =========================
# PATH SETUP (ROBUST)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# =========================
# LOAD MODEL + PREPROCESSOR
# =========================
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("✅ Model Loaded")
print("✅ Preprocessor Loaded")


# =========================
# INPUT SCHEMA
# =========================
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "API running successfully"}


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: Customer):

    try:
        # Convert input → DataFrame
        df = pd.DataFrame([data.dict()])

        # Safety: ensure no missing columns
        df = df.fillna(0)

        # Transform using saved preprocessor
        df = preprocessor.transform(df)

        # Predict
        prediction = model.predict(df)[0]

        # Return clean JSON ONLY
        return {
            "churn": int(prediction),
            "status": "success"
        }

    except Exception as e:
        # NEVER return empty response (fixes Streamlit error)
        return {
            "churn": None,
            "status": "error",
            "message": str(e)
        }