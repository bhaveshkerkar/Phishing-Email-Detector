"""
Phishing Email Detector - FastAPI Backend
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix

from utils.feature_extractor import extract_features, get_red_flags

# --- Load model bundle ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "phishing_model.pkl")
bundle       = joblib.load(MODEL_PATH)
model        = bundle["model"]
tfidf        = bundle["tfidf"]
NUM_FEATURES = bundle["numeric_features"]

# --- App setup ---
app = FastAPI(title="Phishing Email Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow React frontend on any port
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request schema ---
class EmailInput(BaseModel):
    subject: str = ""
    sender:  str = ""
    body:    str = ""
    url:     str = ""


# --- Response helpers ---
def get_risk_level(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def get_risk_color(prob: float) -> str:
    if prob >= 0.80:
        return "#ef4444"   # red
    elif prob >= 0.50:
        return "#f97316"   # orange
    else:
        return "#22c55e"   # green


# --- Main prediction endpoint ---
@app.post("/analyze")
def analyze_email(email: EmailInput):
    # 1. Extract numeric features
    features = extract_features(email.subject, email.sender, email.body, email.url)
    numeric_vec = np.array([[features[f] for f in NUM_FEATURES]])

    # 2. TF-IDF on combined text
    combined_text = f"{email.subject} {email.body}"
    tfidf_vec = tfidf.transform([combined_text])

    # 3. Combine and predict
    X = hstack([tfidf_vec, csr_matrix(numeric_vec)])
    prob        = model.predict_proba(X)[0][1]          # probability of phishing
    prediction  = int(model.predict(X)[0])              # 0 = legit, 1 = phishing

    # 4. Get red flags
    red_flags = get_red_flags(email.subject, email.sender, email.body, email.url)

    return {
        "prediction":  prediction,
        "label":       "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence":  round(float(prob) * 100, 2),
        "risk_level":  get_risk_level(prob),
        "risk_color":  get_risk_color(prob),
        "red_flags":   red_flags,
        "features":    features,
    }


# --- Health check ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Phishing Detector API is running"}


# --- Feature info endpoint ---
@app.get("/features")
def list_features():
    return {"numeric_features": NUM_FEATURES}