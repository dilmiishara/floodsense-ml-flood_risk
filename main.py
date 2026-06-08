import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from datetime import date

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

station_map_lstm = {
    'Ellagawa':   0,
    'Putupaula':  2,
    'Rathnapura': 3
}

station_map_xgb = {
    'Ellagawa':   0,
    'Putupaula':  1,
    'Rathnapura': 2
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

XGB_FEATURES = [
    'station_code', 'rainfall',
    'rainfall_lag_1d', 'rainfall_lag_2d', 'rainfall_lag_3d',
    'rainfall_lag_5d', 'rainfall_lag_7d',
    'rainfall_roll_mean_3d', 'rainfall_roll_mean_5d',
    'rainfall_roll_mean_7d', 'rainfall_roll_mean_14d',
    'rainfall_roll_sum_3d', 'rainfall_roll_sum_5d',
    'rainfall_roll_sum_7d', 'rainfall_roll_sum_14d',
    'rainfall_roll_max_3d', 'rainfall_roll_max_5d',
    'rainfall_roll_max_7d', 'rainfall_roll_max_14d',
    'rainfall_diff_1d', 'rainfall_diff_2d',
    'water_level_lag_1d', 'water_level_lag_2d', 'water_level_lag_3d',
    'month', 'day_of_year', 'humidity', 'temperature',
    'humidity_lag_1d', 'temperature_lag_1d'
]

# ── Load models once on startup ────────────────────────────
print("Loading models...")
lstm_model  = tf.keras.models.load_model('models/flood_risk_model.h5')
lstm_scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
xgb_model   = joblib.load('models/water_level_xgb_model.joblib')
print("All models loaded! Ready!")

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

# Flood Risk schemas
class FloodRiskRequest(BaseModel):
    station:     Literal['Ellagawa', 'Putupaula', 'Rathnapura']
    water_level: float
    rainfall:    float
    humidity:    float
    temperature: float

class FloodRiskResponse(BaseModel):
    station:     str
    risk_level:  str
    risk_code:   int
    confidence:  dict
    water_level: float
    threshold:   dict

# Water Level schemas
class WaterLevelRequest(BaseModel):
    station:       Literal['Ellagawa', 'Putupaula', 'Rathnapura']
    rainfall:      float
    humidity:      float
    temperature:   float
    forecast_date: date

class WaterLevelResponse(BaseModel):
    station:       str
    forecast_date: str
    predicted_wl:  float

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "FloodSense ML API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint 1 — Flood Risk (LSTM)
@app.post("/predict/flood-risk", response_model=FloodRiskResponse)
def predict_flood_risk(req: FloodRiskRequest):
    try:
        sid     = station_map_lstm[req.station]
        wl_norm = normalize_wl(sid, req.water_level)

        single_day        = np.array([[wl_norm, req.water_level,
                                       req.rainfall, req.humidity,
                                       req.temperature]])
        single_day_scaled = lstm_scaler.transform(single_day)
        sequence          = np.tile(single_day_scaled, (SEQ_LEN, 1))
        sequence          = sequence.reshape(1, SEQ_LEN, len(FEATURE_COLS))

        probs = lstm_model.predict(sequence, verbose=0)[0]
        pred  = int(np.argmax(probs))

        return FloodRiskResponse(
            station    = req.station,
            risk_level = risk_labels[pred],
            risk_code  = pred,
            confidence = {
                "Normal":      round(float(probs[0]), 4),
                "Alert":       round(float(probs[1]), 4),
                "Minor Flood": round(float(probs[2]), 4),
                "Major Flood": round(float(probs[3]), 4),
            },
            water_level = req.water_level,
            threshold   = thresholds[sid]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2 — Water Level Forecast (XGBoost)
@app.post("/predict/water-level", response_model=WaterLevelResponse)
def predict_water_level(req: WaterLevelRequest):
    try:
        sid           = station_map_xgb[req.station]
        forecast_date = pd.Timestamp(req.forecast_date)

        row = {
            'station_code':           sid,
            'rainfall':               req.rainfall,
            'rainfall_lag_1d':        req.rainfall,
            'rainfall_lag_2d':        req.rainfall,
            'rainfall_lag_3d':        req.rainfall,
            'rainfall_lag_5d':        req.rainfall,
            'rainfall_lag_7d':        req.rainfall,
            'rainfall_roll_mean_3d':  req.rainfall,
            'rainfall_roll_mean_5d':  req.rainfall,
            'rainfall_roll_mean_7d':  req.rainfall,
            'rainfall_roll_mean_14d': req.rainfall,
            'rainfall_roll_sum_3d':   req.rainfall * 3,
            'rainfall_roll_sum_5d':   req.rainfall * 5,
            'rainfall_roll_sum_7d':   req.rainfall * 7,
            'rainfall_roll_sum_14d':  req.rainfall * 14,
            'rainfall_roll_max_3d':   req.rainfall,
            'rainfall_roll_max_5d':   req.rainfall,
            'rainfall_roll_max_7d':   req.rainfall,
            'rainfall_roll_max_14d':  req.rainfall,
            'rainfall_diff_1d':       0.0,
            'rainfall_diff_2d':       0.0,
            'water_level_lag_1d':     0.0,
            'water_level_lag_2d':     0.0,
            'water_level_lag_3d':     0.0,
            'month':                  forecast_date.month,
            'day_of_year':            forecast_date.day_of_year,
            'humidity':               req.humidity,
            'temperature':            req.temperature,
            'humidity_lag_1d':        req.humidity,
            'temperature_lag_1d':     req.temperature,
        }

        X            = pd.DataFrame([row])[XGB_FEATURES]
        predicted_wl = float(xgb_model.predict(X)[0])

        return WaterLevelResponse(
            station       = req.station,
            forecast_date = str(req.forecast_date),
            predicted_wl  = round(predicted_wl, 3)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))