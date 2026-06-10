import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from datetime import date, timedelta
from psycopg2 import pool as psycopg2_pool
import psycopg2
import psycopg2.extras
import os

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

# ── Affected Area Lookup Table ─────────────────────────────
def get_duration_factor(days: float) -> float:
    if days < 1:
        return 1.0
    elif days < 2:
        return 1.2
    elif days < 3:
        return 1.4
    elif days < 5:
        return 1.6
    else:
        return 1.8

AFFECTED_AREA_LOOKUP = {
    (0, 0, 0): 0.0,
    (1, 0, 0): 1.0,
    (0, 1, 0): 1.0,
    (0, 0, 1): 1.5,
    (1, 1, 0): 2.0,
    (1, 0, 1): 2.5,
    (0, 1, 1): 2.5,
    (1, 1, 1): 3.0,
    (2, 0, 0): 2.0,
    (0, 2, 0): 2.0,
    (0, 0, 2): 3.0,
    (2, 1, 0): 3.0,
    (2, 0, 1): 3.5,
    (0, 2, 1): 3.5,
    (1, 2, 0): 3.0,
    (0, 1, 2): 4.0,
    (1, 0, 2): 4.0,
    (2, 2, 0): 4.0,
    (2, 0, 2): 4.8,
    (0, 2, 2): 4.8,
    (2, 2, 2): 5.5,
    (3, 0, 0): 3.0,
    (0, 3, 0): 3.0,
    (0, 0, 3): 5.0,
    (3, 1, 0): 4.0,
    (3, 0, 1): 4.5,
    (0, 3, 1): 4.5,
    (1, 3, 0): 4.0,
    (0, 1, 3): 6.0,
    (1, 0, 3): 6.0,
    (3, 2, 0): 5.0,
    (3, 0, 2): 6.0,
    (0, 3, 2): 6.0,
    (2, 3, 0): 5.0,
    (0, 2, 3): 7.0,
    (2, 0, 3): 7.0,
    (3, 3, 0): 6.0,
    (3, 0, 3): 8.0,
    (0, 3, 3): 8.0,
    (3, 3, 2): 9.0,
    (3, 2, 3): 11.0,
    (2, 3, 3): 11.0,
    (3, 3, 1): 7.0,
    (3, 1, 3): 10.0,
    (1, 3, 3): 10.0,
    (3, 2, 2): 7.4,
    (2, 3, 2): 7.4,
    (2, 2, 3): 9.0,
    (3, 3, 3): 14.9,
    (3, 2, 1): 6.0,
    (3, 1, 2): 7.0,
    (2, 3, 1): 6.0,
    (1, 3, 2): 7.0,
    (2, 1, 3): 8.0,
    (1, 2, 3): 8.0,
    (2, 2, 1): 4.5,
    (2, 1, 2): 5.0,
    (1, 2, 2): 5.0,
    (1, 1, 2): 4.0,
    (1, 2, 1): 3.5,
    (2, 1, 1): 3.5,
    (1, 1, 3): 7.0,
    (1, 3, 1): 5.0,
    (3, 1, 1): 5.0,
}

def get_severity_label(area: float) -> str:
    if area == 0:
        return "Normal"
    elif area < 3.0:
        return "Alert"
    elif area < 7.4:
        return "Minor Flood"
    else:
        return "Major Flood"

# ── Models (loaded at startup) ─────────────────────────────
lstm_model  = None
lstm_scaler = None
xgb_model   = None

# ── Connection pool (created at startup) ──────────────────
db_pool = None

# ── Startup & Shutdown ─────────────────────────────────────
@app.on_event("startup")
def startup():
    global db_pool, lstm_model, lstm_scaler, xgb_model

    # Load models
    print("Loading models...")
    lstm_model  = tf.keras.models.load_model('models/flood_risk_model.h5')
    lstm_scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
    xgb_model   = joblib.load('models/water_level_xgb_model.joblib')
    print("All models loaded!")

    # Create connection pool
    db_pool = psycopg2_pool.SimpleConnectionPool(
        minconn  = 1,
        maxconn  = 5,
        host     = os.getenv("DB_HOST",     "aws-1-ap-south-1.pooler.supabase.com"),
        port     = int(os.getenv("DB_PORT", "6543")),
        database = os.getenv("DB_NAME",     "postgres"),
        user     = os.getenv("DB_USER",     "postgres.jxwytxoedkdrcelrvixf"),
        password = os.getenv("DB_PASSWORD", "Znd@l78P2021"),
        sslmode  = "require"
    )
    print("✅ Database connection pool created!")
    print("✅ FloodSense ML API Ready!")


@app.on_event("shutdown")
def shutdown():
    global db_pool
    if db_pool:
        db_pool.closeall()
        print("✅ Database connection pool closed!")


# ── Database helpers ───────────────────────────────────────
def get_db_connection():
    return db_pool.getconn()

def release_db_connection(conn):
    db_pool.putconn(conn)


# ── Fetch all past data in ONE db call ─────────────────────
def fetch_past_data(station: str, forecast_date: date) -> tuple:
    date_from_14 = forecast_date - timedelta(days=14)
    date_from_2  = forecast_date - timedelta(days=2)

    wl_query = """
        SELECT
            DATE(recorded_at AT TIME ZONE 'Asia/Colombo') AS date,
            AVG(water_level)  AS water_level,
            SUM(rainfall_mm)  AS rainfall
        FROM water_level_logs
        WHERE station_name = %s
          AND recorded_at >= %s
          AND recorded_at < %s
        GROUP BY DATE(recorded_at AT TIME ZONE 'Asia/Colombo')
        ORDER BY date ASC
    """

    weather_query = """
        SELECT
            DATE(recorded_at AT TIME ZONE 'Asia/Colombo') AS date,
            AVG(humidity) AS humidity,
            AVG(temp_c)   AS temperature
        FROM weather_logs
        WHERE recorded_at >= %s
          AND recorded_at < %s
        GROUP BY DATE(recorded_at AT TIME ZONE 'Asia/Colombo')
        ORDER BY date ASC
    """

    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(wl_query, (station, date_from_14, forecast_date))
        wl_rows = cursor.fetchall()

        cursor.execute(weather_query, (date_from_2, forecast_date))
        weather_rows = cursor.fetchall()

        cursor.close()

        wl_df = pd.DataFrame(wl_rows)
        if not wl_df.empty:
            wl_df['date']        = pd.to_datetime(wl_df['date'])
            wl_df['water_level'] = pd.to_numeric(wl_df['water_level'], errors='coerce')
            wl_df['rainfall']    = pd.to_numeric(wl_df['rainfall'],    errors='coerce')

        weather_df = pd.DataFrame(weather_rows)
        if not weather_df.empty:
            weather_df['date']        = pd.to_datetime(weather_df['date'])
            weather_df['humidity']    = pd.to_numeric(weather_df['humidity'],    errors='coerce')
            weather_df['temperature'] = pd.to_numeric(weather_df['temperature'], errors='coerce')

        return wl_df, weather_df

    except Exception as e:
        print(f"DB error fetching past data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if conn:
            release_db_connection(conn)


# ── Build XGBoost feature row ──────────────────────────────
def build_xgb_features(
    station:       str,
    rainfall:      float,
    humidity:      float,
    temperature:   float,
    forecast_date: date
) -> pd.DataFrame:

    sid         = station_map_xgb[station]
    forecast_ts = pd.Timestamp(forecast_date)

    # Step 1 & 2: Fetch all past data in one DB call
    wl_df, weather_df = fetch_past_data(station, forecast_date)

    # Step 3: Append current day to water level history
    current_row = pd.DataFrame([{
        'date'       : forecast_ts,
        'water_level': np.nan,
        'rainfall'   : float(rainfall)
    }])

    if not wl_df.empty:
        history = pd.concat([wl_df, current_row], ignore_index=True)
    else:
        history = current_row.copy()

    history['date']        = pd.to_datetime(history['date'])
    history['rainfall']    = pd.to_numeric(history['rainfall'],    errors='coerce')
    history['water_level'] = pd.to_numeric(history['water_level'], errors='coerce')
    history = history.sort_values('date').reset_index(drop=True)

    # Step 4: Rainfall lag
    def get_rainfall_lag(days: int) -> float:
        target = forecast_ts - pd.Timedelta(days=days)
        mask   = history['date'] == target
        if mask.any():
            val = history.loc[mask, 'rainfall'].values[0]
            if pd.notna(val):
                return float(val)
        return float(rainfall)

    # Step 5: Water level lag
    def get_wl_lag(days: int) -> float:
        target = forecast_ts - pd.Timedelta(days=days)
        mask   = history['date'] == target
        if mask.any():
            val = history.loc[mask, 'water_level'].values[0]
            if pd.notna(val):
                return float(val)
        return 0.0

    # Step 6: Rolling rainfall
    def get_rolling(window: int, func: str) -> float:
        mask = (
            (history['date'] < forecast_ts) &
            (history['date'] >= forecast_ts - pd.Timedelta(days=window))
        )
        past = history.loc[mask, 'rainfall'].dropna()
        if past.empty:
            return float(rainfall)
        if func == 'mean': return float(past.mean())
        if func == 'sum':  return float(past.sum())
        if func == 'max':  return float(past.max())
        return float(rainfall)

    # Step 7: Weather lag
    def get_weather_lag(col: str, fallback: float) -> float:
        if weather_df.empty:
            return float(fallback)
        wdf         = weather_df.copy()
        wdf['date'] = pd.to_datetime(wdf['date'])
        target      = forecast_ts - pd.Timedelta(days=1)
        mask        = wdf['date'] == target
        if mask.any():
            val = wdf.loc[mask, col].values[0]
            if pd.notna(val):
                return float(val)
        return float(fallback)

    # Step 8: Rainfall rate of change
    r_lag1  = get_rainfall_lag(1)
    r_lag2  = get_rainfall_lag(2)
    r_diff1 = float(rainfall) - r_lag1
    r_diff2 = float(rainfall) - r_lag2

    # Step 9: Assemble feature row
    row = {
        'station_code'          : int(sid),
        'rainfall'              : float(rainfall),
        'rainfall_lag_1d'       : r_lag1,
        'rainfall_lag_2d'       : r_lag2,
        'rainfall_lag_3d'       : get_rainfall_lag(3),
        'rainfall_lag_5d'       : get_rainfall_lag(5),
        'rainfall_lag_7d'       : get_rainfall_lag(7),
        'rainfall_roll_mean_3d' : get_rolling(3,  'mean'),
        'rainfall_roll_mean_5d' : get_rolling(5,  'mean'),
        'rainfall_roll_mean_7d' : get_rolling(7,  'mean'),
        'rainfall_roll_mean_14d': get_rolling(14, 'mean'),
        'rainfall_roll_sum_3d'  : get_rolling(3,  'sum'),
        'rainfall_roll_sum_5d'  : get_rolling(5,  'sum'),
        'rainfall_roll_sum_7d'  : get_rolling(7,  'sum'),
        'rainfall_roll_sum_14d' : get_rolling(14, 'sum'),
        'rainfall_roll_max_3d'  : get_rolling(3,  'max'),
        'rainfall_roll_max_5d'  : get_rolling(5,  'max'),
        'rainfall_roll_max_7d'  : get_rolling(7,  'max'),
        'rainfall_roll_max_14d' : get_rolling(14, 'max'),
        'rainfall_diff_1d'      : r_diff1,
        'rainfall_diff_2d'      : r_diff2,
        'water_level_lag_1d'    : get_wl_lag(1),
        'water_level_lag_2d'    : get_wl_lag(2),
        'water_level_lag_3d'    : get_wl_lag(3),
        'month'                 : int(forecast_ts.month),
        'day_of_year'           : int(forecast_ts.day_of_year),
        'humidity'              : float(humidity),
        'temperature'           : float(temperature),
        'humidity_lag_1d'       : get_weather_lag('humidity',    humidity),
        'temperature_lag_1d'    : get_weather_lag('temperature', temperature),
    }

    return pd.DataFrame([row])[XGB_FEATURES]


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
        X = build_xgb_features(
            station       = req.station,
            rainfall      = req.rainfall,
            humidity      = req.humidity,
            temperature   = req.temperature,
            forecast_date = req.forecast_date
        )

        predicted_wl = float(xgb_model.predict(X)[0])

        return WaterLevelResponse(
            station       = req.station,
            forecast_date = str(req.forecast_date),
            predicted_wl  = round(predicted_wl, 3)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Test endpoint (remove before final deployment) ─────────
@app.get("/test-db")
def test_db():
    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM water_level_logs")
        wl_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM weather_logs")
        weather_count = cursor.fetchone()[0]
        cursor.close()
        return {
            "status"          : "connected",
            "water_level_logs": int(wl_count),
            "weather_logs"    : int(weather_count)
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        if conn:
            release_db_connection(conn)