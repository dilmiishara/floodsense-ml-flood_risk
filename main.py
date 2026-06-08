import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="FloodSense ML API")

# ── Constants ──────────────────────────────────────────────
SEQ_LEN = 7
FEATURE_COLS = [
    'water_level_norm',
    'water_level',
    'rainfall',
    'humidity',
    'temperature'
]

station_map = {
    'Ellagawa':   0,
    'Putupaula':  2,
    'Rathnapura': 3
}

thresholds = {
    0: {'alert': 10.00, 'minor': 10.70, 'major': 12.20},
    2: {'alert':  3.00, 'minor':  4.00, 'major':  5.00},
    3: {'alert':  5.20, 'minor':  7.50, 'major':  9.50},
}

risk_labels = {
    0: 'Normal',
    1: 'Alert',
    2: 'Minor Flood',
    3: 'Major Flood'
}

# ── Load model and scaler once on startup ──────────────────
print("Loading model and scaler...")
model = tf.keras.models.load_model('models/flood_risk_model.h5')
scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
print("Ready!")

# ── Helper functions ───────────────────────────────────────
def normalize_wl(station_id: int, wl: float) -> float:
    t = thresholds[station_id]
    if wl >= t['major']:
        return 3.0 + (wl - t['major']) / (t['major'] - t['minor'] + 1e-8)
    elif wl >= t['minor']:
        return 2.0 + (wl - t['minor']) / (t['major'] - t['minor'] + 1e-8)
    elif wl >= t['alert']:
        return 1.0 + (wl - t['alert']) / (t['minor'] - t['alert'] + 1e-8)
    else:
        return wl / t['alert']

# ── Request / Response schemas ─────────────────────────────
class FloodRiskRequest(BaseModel):
    station:     Literal['Ellagawa', 'Putupaula', 'Rathnapura']
    water_level: float
    rainfall:    float
    humidity:    float
    temperature: float

class FloodRiskResponse(BaseModel):
    station:      str
    risk_level:   str
    risk_code:    int
    confidence:   dict
    water_level:  float
    threshold:    dict

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "FloodSense ML API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/flood-risk", response_model=FloodRiskResponse)
def predict_flood_risk(req: FloodRiskRequest):
    try:
        sid     = station_map[req.station]
        wl_norm = normalize_wl(sid, req.water_level)

        # Build scaled sequence (repeat single reading 7 times)
        single_day        = np.array([[wl_norm, req.water_level,
                                       req.rainfall, req.humidity,
                                       req.temperature]])
        single_day_scaled = scaler.transform(single_day)
        sequence          = np.tile(single_day_scaled, (SEQ_LEN, 1))
        sequence          = sequence.reshape(1, SEQ_LEN, len(FEATURE_COLS))

        # Predict
        probs = model.predict(sequence, verbose=0)[0]
        pred  = int(np.argmax(probs))

        return FloodRiskResponse(
            station    = req.station,
            risk_level = risk_labels[pred],
            risk_code  = pred,
            confidence = {
                "Normal":     round(float(probs[0]), 4),
                "Alert":      round(float(probs[1]), 4),
                "Minor Flood": round(float(probs[2]), 4),
                "Major Flood": round(float(probs[3]), 4),
            },
            water_level = req.water_level,
            threshold   = thresholds[sid]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))