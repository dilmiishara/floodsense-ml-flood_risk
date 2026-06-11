"""
FloodSense Prediction Pipeline
--------------------------------
1. Fetch 12-hour ahead weather from Open-Meteo
2. Run Model 2 (Water Level Forecast - XGBoost)
3. Run Model 1 (Flood Risk - LSTM)
4. Calculate affected area per station
5. Save results to predictions table
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import psycopg2
import psycopg2.extras
from psycopg2 import pool as psycopg2_pool
import urllib3
import os
import time
from datetime import datetime, timedelta, date

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Import shared functions from main.py ───────────────────
from main import (
    SEQ_LEN,
    FEATURE_COLS,
    station_map_lstm,
    station_map_xgb,
    thresholds,
    risk_labels,
    XGB_FEATURES,
    STATION_THRESHOLDS,
    normalize_wl,
    build_xgb_features,
    get_flood_duration_days,
)

import main as main_module

# ── Database Config ────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "aws-1-ap-south-1.pooler.supabase.com"),
    "port":     int(os.getenv("DB_PORT", "6543")),
    "database": os.getenv("DB_NAME",     "postgres"),
    "user":     os.getenv("DB_USER",     "postgres.jxwytxoedkdrcelrvixf"),
    "password": os.getenv("DB_PASSWORD", "Znd@l78P2021"),
    "sslmode":  "require"
}

# ── Only initialize DB when running directly ───────────────
if __name__ == "__main__":
    print("Connecting to database...")
    try:
        main_module.db_pool = psycopg2_pool.SimpleConnectionPool(
            minconn  = 1,
            maxconn  = 5,
            host     = DB_CONFIG["host"],
            port     = DB_CONFIG["port"],
            database = DB_CONFIG["database"],
            user     = DB_CONFIG["user"],
            password = DB_CONFIG["password"],
            sslmode  = DB_CONFIG["sslmode"]
        )
        print("✅ Database connected!")
    except Exception as e:
        print(f"⚠️ DB connection failed: {e}")
        main_module.db_pool = None

    # Load models when running directly
    print("Loading models...")
    main_module.lstm_model  = tf.keras.models.load_model('models/flood_risk_model.h5')
    main_module.lstm_scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
    main_module.xgb_model   = joblib.load('models/water_level_xgb_model.joblib')
    print("✅ Models loaded!")


# ── Station Coordinates ────────────────────────────────────
STATIONS = {
    "Ellagawa":   {"latitude": 6.7320, "longitude": 80.2107},
    "Putupaula":  {"latitude": 6.6014, "longitude": 80.0564},
    "Rathnapura": {"latitude": 6.6828, "longitude": 80.3992},
}

# ── Per Station Affected Area Table ───────────────────────
STATION_AREA = {
    'Ellagawa':   {0: 0.0, 1: 0.5, 2: 1.5, 3: 3.5},
    'Putupaula':  {0: 0.0, 1: 0.5, 2: 1.5, 3: 3.0},
    'Rathnapura': {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.4},
}

# ── Duration Factor ────────────────────────────────────────
def get_duration_factor(days: float) -> float:
    if days < 1:   return 1.0
    elif days < 2: return 1.2
    elif days < 3: return 1.4
    elif days < 5: return 1.6
    else:          return 1.8


# ── Step 1: Fetch Weather Forecast from Open-Meteo ────────
def fetch_weather_forecast(station_name: str, target_time: datetime) -> dict:
    station = STATIONS[station_name]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={station['latitude']}"
        f"&longitude={station['longitude']}"
        f"&hourly=temperature_2m,relativehumidity_2m,precipitation"
        f"&timezone=Asia/Colombo"
        f"&forecast_days=2"
    )

    print(f"  Fetching weather for {station_name}...")

    # Retry up to 3 times
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=15, verify=False)
            response.raise_for_status()
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                print(f"  ⚠️ Attempt {attempt+1} failed, retrying in 3s...")
                time.sleep(3)
            else:
                raise last_error

    data       = response.json()
    target_str = target_time.strftime("%Y-%m-%dT%H:00")
    times      = data["hourly"]["time"]
    index      = times.index(target_str) if target_str in times else 12

    temperature = data["hourly"]["temperature_2m"][index]
    humidity    = data["hourly"]["relativehumidity_2m"][index]
    rainfall    = data["hourly"]["precipitation"][index]

    print(f"  {station_name} @ {target_str}:")
    print(f"    Temp={temperature}°C  Humidity={humidity}%  Rain={rainfall}mm")

    return {
        "temperature":   temperature,
        "humidity":      humidity,
        "rainfall":      rainfall,
        "forecast_time": target_str
    }


# ── Step 2: Run Model 2 (Water Level) ─────────────────────
def get_predicted_water_level(station_name, rainfall, humidity, temperature, forecast_date):
    print(f"  Running water level model for {station_name}...")
    X = build_xgb_features(
        station       = station_name,
        rainfall      = rainfall,
        humidity      = humidity,
        temperature   = temperature,
        forecast_date = forecast_date
    )
    predicted_wl = float(main_module.xgb_model.predict(X)[0])
    print(f"  {station_name} predicted water level: {predicted_wl}m")
    return round(predicted_wl, 3)


# ── Step 3: Run Model 1 (Flood Risk) ──────────────────────
def get_flood_risk(station_name, water_level, rainfall, humidity, temperature):
    print(f"  Running flood risk model for {station_name}...")
    sid     = station_map_lstm[station_name]
    wl_norm = normalize_wl(sid, water_level)

    single_day        = np.array([[wl_norm, water_level, rainfall, humidity, temperature]])
    single_day_scaled = main_module.lstm_scaler.transform(single_day)
    sequence          = np.tile(single_day_scaled, (SEQ_LEN, 1))
    sequence          = sequence.reshape(1, SEQ_LEN, len(FEATURE_COLS))

    probs      = main_module.lstm_model.predict(sequence, verbose=0)[0]
    pred       = int(np.argmax(probs))
    risk_level = risk_labels[pred]

    print(f"  {station_name} flood risk: {risk_level}")
    return {"risk_level": risk_level, "risk_code": pred}


# ── Step 4: Calculate Affected Area ───────────────────────
def calculate_affected_area(station_name, risk_code, duration_days):
    base_area       = STATION_AREA[station_name][risk_code]
    duration_factor = get_duration_factor(duration_days)
    final_area      = round(base_area * duration_factor, 2)
    print(f"  {station_name} area: {base_area} x {duration_factor} = {final_area} sq km")
    return final_area


# ── Step 5: Save Results To Database ──────────────────────
def save_predictions(results: list):
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO public.predictions (
                station_name, forecast_time, predicted_water_level,
                flood_risk_level, affected_area_sqkm, duration_days,
                temperature, humidity, rainfall, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        for result in results:
            cursor.execute(insert_query, (
                result["station_name"],
                result["forecast_time"],
                result["predicted_water_level"],
                result["flood_risk_level"],
                result["affected_area_sqkm"],
                result["duration_days"],
                result["temperature"],
                result["humidity"],
                result["rainfall"]
            ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"\n✅ Saved {len(results)} predictions to database!")
    except Exception as e:
        print(f"\n❌ Database save error: {e}")
        raise


# ── Main Pipeline Function ─────────────────────────────────
def run_pipeline():
    print("\n" + "="*50)
    print("FloodSense Prediction Pipeline Starting...")
    print("="*50)

    now           = datetime.now()
    target_time   = now + timedelta(hours=12)
    forecast_date = target_time.date()

    print(f"\nCurrent time:  {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Forecast time: {target_time.strftime('%Y-%m-%d %H:%M')} (+12 hours)")

    results = []

    for station_name in ["Ellagawa", "Putupaula", "Rathnapura"]:
        print(f"\n── Processing {station_name} ──────────────")
        time.sleep(2)  # avoid rate limiting

        try:
            weather      = fetch_weather_forecast(station_name, target_time)
            predicted_wl = get_predicted_water_level(
                station_name  = station_name,
                rainfall      = weather["rainfall"],
                humidity      = weather["humidity"],
                temperature   = weather["temperature"],
                forecast_date = forecast_date
            )
            risk = get_flood_risk(
                station_name = station_name,
                water_level  = predicted_wl,
                rainfall     = weather["rainfall"],
                humidity     = weather["humidity"],
                temperature  = weather["temperature"]
            )
            duration_days = get_flood_duration_days(
                station      = station_name,
                current_risk = risk["risk_level"]
            )
            print(f"  {station_name} flood duration: {duration_days} days")

            affected_area = calculate_affected_area(
                station_name  = station_name,
                risk_code     = risk["risk_code"],
                duration_days = duration_days
            )

            results.append({
                "station_name":          station_name,
                "forecast_time":         target_time,
                "predicted_water_level": predicted_wl,
                "flood_risk_level":      risk["risk_level"],
                "affected_area_sqkm":    affected_area,
                "duration_days":         duration_days,
                "temperature":           weather["temperature"],
                "humidity":              weather["humidity"],
                "rainfall":              weather["rainfall"]
            })

        except Exception as e:
            print(f"  ❌ Error processing {station_name}: {e}")
            continue

    if results:
        save_predictions(results)
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        for r in results:
            print(f"\n{r['station_name']}:")
            print(f"  Forecast Time:  {r['forecast_time']}")
            print(f"  Water Level:    {r['predicted_water_level']}m")
            print(f"  Flood Risk:     {r['flood_risk_level']}")
            print(f"  Affected Area:  {r['affected_area_sqkm']} sq km")
            print(f"  Duration:       {r['duration_days']} days")
        print("\n" + "="*50)
    else:
        print("\n❌ No results to save!")

    return results


# ── Run directly for testing ───────────────────────────────
if __name__ == "__main__":
    run_pipeline()