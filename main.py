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

# ── Station thresholds by name (used by pipeline.py too) ──
STATION_THRESHOLDS = {
    'Ellagawa':   {'alert': 10.00, 'minor': 10.70, 'major': 12.20},
    'Putupaula':  {'alert':  3.00, 'minor':  4.00, 'major':  5.00},
    'Rathnapura': {'alert':  5.20, 'minor':  7.50, 'major':  9.50},
}

# ── Models ─────────────────────────────────────────────────
lstm_model  = None
lstm_scaler = None
xgb_model   = None

# ── Connection pool ────────────────────────────────────────
db_pool = None

# ── Startup & Shutdown ─────────────────────────────────────
@app.on_event("startup")
def startup():
    global db_pool, lstm_model, lstm_scaler, xgb_model

    print("Loading models...")
    lstm_model  = tf.keras.models.load_model('models/flood_risk_model.h5')
    lstm_scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
    xgb_model   = joblib.load('models/water_level_xgb_model.joblib')
    print("All models loaded!")

    try:
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
        print("Database connection pool created!")
    except Exception as e:
        print(f"Database connection failed: {e} - continuing without DB")
        db_pool = None

    print("FloodSense ML API Ready!")


@app.on_event("shutdown")
def shutdown():
    global db_pool
    if db_pool:
        db_pool.closeall()


# ── Database helpers ───────────────────────────────────────
def get_db_connection():
    return db_pool.getconn()

def release_db_connection(conn):
    db_pool.putconn(conn)


# ── Fetch past water level + weather data ─────────────────
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

    wl_df, weather_df = fetch_past_data(station, forecast_date)

    current_row = pd.DataFrame([{
        'date'       : forecast_ts,
        'water_level': np.nan,
        'rainfall'   : float(rainfall)
    }])

    history = pd.concat([wl_df, current_row], ignore_index=True) if not wl_df.empty else current_row.copy()
    history['date']        = pd.to_datetime(history['date'])
    history['rainfall']    = pd.to_numeric(history['rainfall'],    errors='coerce')
    history['water_level'] = pd.to_numeric(history['water_level'], errors='coerce')
    history = history.sort_values('date').reset_index(drop=True)

    def get_rainfall_lag(days: int) -> float:
        target = forecast_ts - pd.Timedelta(days=days)
        mask   = history['date'] == target
        if mask.any():
            val = history.loc[mask, 'rainfall'].values[0]
            if pd.notna(val): return float(val)
        return float(rainfall)

    def get_wl_lag(days: int) -> float:
        target = forecast_ts - pd.Timedelta(days=days)
        mask   = history['date'] == target
        if mask.any():
            val = history.loc[mask, 'water_level'].values[0]
            if pd.notna(val): return float(val)
        return 0.0

    def get_rolling(window: int, func: str) -> float:
        mask = (
            (history['date'] < forecast_ts) &
            (history['date'] >= forecast_ts - pd.Timedelta(days=window))
        )
        past = history.loc[mask, 'rainfall'].dropna()
        if past.empty: return float(rainfall)
        if func == 'mean': return float(past.mean())
        if func == 'sum':  return float(past.sum())
        if func == 'max':  return float(past.max())
        return float(rainfall)

    def get_weather_lag(col: str, fallback: float) -> float:
        if weather_df.empty: return float(fallback)
        wdf         = weather_df.copy()
        wdf['date'] = pd.to_datetime(wdf['date'])
        target      = forecast_ts - pd.Timedelta(days=1)
        mask        = wdf['date'] == target
        if mask.any():
            val = wdf.loc[mask, col].values[0]
            if pd.notna(val): return float(val)
        return float(fallback)

    r_lag1 = get_rainfall_lag(1)
    r_lag2 = get_rainfall_lag(2)

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
        'rainfall_diff_1d'      : float(rainfall) - r_lag1,
        'rainfall_diff_2d'      : float(rainfall) - r_lag2,
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


# ── Normalize water level against thresholds ──────────────
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


# ── Get flood duration from DB ─────────────────────────────
def get_flood_duration_days(station: str, current_risk: str) -> float:
    if current_risk == 'Normal':
        return 0.0
    if db_pool is None:
        return 0.5

    min_wl = STATION_THRESHOLDS[station]['alert']
    conn   = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""
            SELECT recorded_at, water_level
            FROM water_level_logs
            WHERE station_name = %s
              AND recorded_at >= NOW() - INTERVAL '10 days'
              AND water_level IS NOT NULL
            ORDER BY recorded_at DESC
        """, (station,))
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            return 0.5

        consecutive_count = 0
        for row in rows:
            if float(row['water_level']) >= min_wl:
                consecutive_count += 1
            else:
                break

        if consecutive_count == 0:
            return 0.5

        oldest        = rows[consecutive_count - 1]['recorded_at']
        newest        = rows[0]['recorded_at']
        duration_days = (newest - oldest).total_seconds() / 86400.0
        return round(max(duration_days, 0.5), 2)

    except Exception as e:
        print(f"Duration check error for {station}: {e}")
        return 0.5
    finally:
        if conn:
            release_db_connection(conn)


# ── Request / Response schemas ─────────────────────────────

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


# Endpoint 1 — Flood Risk Classification (LSTM)
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


# Endpoint 3 — Pipeline trigger (called by Cloud Scheduler)
@app.post("/pipeline/run")
def trigger_pipeline():
    try:
        from pipeline import run_pipeline
        results = run_pipeline()
        return {
            "status":  "success",
            "message": f"Saved {len(results)} predictions",
            "results": [
                {
                    "station":       r["station_name"],
                    "risk_level":    r["flood_risk_level"],
                    "affected_area": r["affected_area_sqkm"]
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 4 — Test DB connection
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