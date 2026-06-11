"""
FloodSense Prediction Pipeline
--------------------------------
This script imports functions directly from main.py
and runs the complete prediction pipeline:

1. Load models directly
2. Fetch 12-hour ahead weather from Open-Meteo
3. Run Model 2 (Water Level Forecast)
4. Run Model 1 (Flood Risk) using predicted water level
5. Calculate affected area per station
6. Save results to predictions table
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import psycopg2
import psycopg2.extras
from psycopg2 import pool as psycopg2_pool
import os
from datetime import datetime, timedelta, date

# ── Import shared functions and constants from main.py ─────
from main import (
    # Constants
    SEQ_LEN,
    FEATURE_COLS,
    station_map_lstm,
    station_map_xgb,
    thresholds,
    risk_labels,
    XGB_FEATURES,
    STATION_THRESHOLDS,

    # Helper functions
    normalize_wl,
    build_xgb_features,
    get_flood_duration_days,
)

# ── Import main module to set db_pool ─────────────────────
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

# ── Initialize DB pool for pipeline ───────────────────────
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

# ── Load Models Directly ───────────────────────────────────
print("Loading models...")
lstm_model  = tf.keras.models.load_model('models/flood_risk_model.h5')
lstm_scaler = joblib.load('models/floodsense_lstm_scaler.pkl')
xgb_model   = joblib.load('models/water_level_xgb_model.joblib')
print("✅ Models loaded!")

# ── Station Coordinates ────────────────────────────────────
STATIONS = {
    "Ellagawa": {
        "latitude":  6.7320,
        "longitude": 80.2107,
    },
    "Putupaula": {
        "latitude":  6.6014,
        "longitude": 80.0564,
    },
    "Rathnapura": {
        "latitude":  6.6828,
        "longitude": 80.3992,
    }
}

# ── Per Station Affected Area Table ───────────────────────
# Risk codes: 0=Normal, 1=Alert, 2=Minor Flood, 3=Major Flood
# Values based on:
#   - DMC/ReliefWeb 2017 satellite data
#   - Rathnapura Municipal Council flood study
STATION_AREA = {
    'Ellagawa': {
        0: 0.0,   # Normal
        1: 0.5,   # Alert
        2: 1.5,   # Minor Flood
        3: 3.5,   # Major Flood
    },
    'Putupaula': {
        0: 0.0,
        1: 0.5,
        2: 1.5,
        3: 3.0,
    },
    'Rathnapura': {
        0: 0.0,
        1: 1.0,
        2: 3.0,
        3: 7.4,   # from MC flood study (737 ha)
    },
}

# ── Duration Factor ────────────────────────────────────────
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


# ── Step 1: Fetch Weather Forecast from Open-Meteo ────────
def fetch_weather_forecast(station_name: str, target_time: datetime) -> dict:
    """
    Fetch hourly weather forecast from Open-Meteo API
    for a specific station at a specific future time.
    """
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
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Find the target hour in the response
    target_str = target_time.strftime("%Y-%m-%dT%H:00")
    times      = data["hourly"]["time"]

    if target_str not in times:
        print(f"  ⚠️ Target time {target_str} not found, using index 12")
        index = 12  # fallback
    else:
        index = times.index(target_str)

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


# ── Step 2: Run Model 2 (Water Level Forecast) ────────────
def get_predicted_water_level(
    station_name:  str,
    rainfall:      float,
    humidity:      float,
    temperature:   float,
    forecast_date: date
) -> float:
    """
    Run XGBoost water level model directly.
    Returns predicted water level in meters.
    """
    print(f"  Running water level model for {station_name}...")

    X = build_xgb_features(
        station       = station_name,
        rainfall      = rainfall,
        humidity      = humidity,
        temperature   = temperature,
        forecast_date = forecast_date
    )

    predicted_wl = float(xgb_model.predict(X)[0])
    print(f"  {station_name} predicted water level: {predicted_wl}m")
    return round(predicted_wl, 3)


# ── Step 3: Run Model 1 (Flood Risk) ──────────────────────
def get_flood_risk(
    station_name: str,
    water_level:  float,
    rainfall:     float,
    humidity:     float,
    temperature:  float
) -> dict:
    """
    Run LSTM flood risk model directly.
    Returns flood risk level and code.
    """
    print(f"  Running flood risk model for {station_name}...")

    sid     = station_map_lstm[station_name]
    wl_norm = normalize_wl(sid, water_level)

    single_day        = np.array([[wl_norm, water_level,
                                   rainfall, humidity,
                                   temperature]])
    single_day_scaled = lstm_scaler.transform(single_day)
    sequence          = np.tile(single_day_scaled, (SEQ_LEN, 1))
    sequence          = sequence.reshape(1, SEQ_LEN, len(FEATURE_COLS))

    probs = lstm_model.predict(sequence, verbose=0)[0]
    pred  = int(np.argmax(probs))

    risk_level = risk_labels[pred]
    print(f"  {station_name} flood risk: {risk_level}")

    return {
        "risk_level": risk_level,
        "risk_code":  pred
    }


# ── Step 4: Calculate Affected Area ───────────────────────
def calculate_affected_area(
    station_name:  str,
    risk_code:     int,
    duration_days: float
) -> float:
    """
    Look up base affected area and apply duration factor.
    """
    base_area       = STATION_AREA[station_name][risk_code]
    duration_factor = get_duration_factor(duration_days)
    final_area      = round(base_area * duration_factor, 2)

    print(f"  {station_name} area: {base_area} x {duration_factor} (duration factor) = {final_area} sq km")
    return final_area


# ── Step 5: Save Results To Database ──────────────────────
def save_predictions(results: list):
    """
    Save 3 rows (one per station) to predictions table.
    """
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO public.predictions (
                station_name,
                forecast_time,
                predicted_water_level,
                flood_risk_level,
                affected_area_sqkm,
                duration_days,
                temperature,
                humidity,
                rainfall,
                created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
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
    """
    Main function - runs the complete prediction pipeline.
    Call this from scheduler or manually for testing.
    """
    print("\n" + "="*50)
    print("FloodSense Prediction Pipeline Starting...")
    print("="*50)

    # Calculate target forecast time (12 hours from now)
    now           = datetime.now()
    target_time   = now + timedelta(hours=12)
    forecast_date = target_time.date()

    print(f"\nCurrent time:  {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Forecast time: {target_time.strftime('%Y-%m-%d %H:%M')} (+12 hours)")

    results = []

    # Process each station one by one
    for station_name in ["Ellagawa", "Putupaula", "Rathnapura"]:
        print(f"\n── Processing {station_name} ──────────────")

        try:
            # Step 1 — Get weather forecast from Open-Meteo
            weather = fetch_weather_forecast(station_name, target_time)

            # Step 2 — Get predicted water level (Model 2 - XGBoost)
            predicted_wl = get_predicted_water_level(
                station_name  = station_name,
                rainfall      = weather["rainfall"],
                humidity      = weather["humidity"],
                temperature   = weather["temperature"],
                forecast_date = forecast_date
            )

            # Step 3 — Get flood risk using predicted water level (Model 1 - LSTM)
            risk = get_flood_risk(
                station_name = station_name,
                water_level  = predicted_wl,
                rainfall     = weather["rainfall"],
                humidity     = weather["humidity"],
                temperature  = weather["temperature"]
            )

            # Step 4 — Get flood duration from database
            duration_days = get_flood_duration_days(
                station      = station_name,
                current_risk = risk["risk_level"]
            )
            print(f"  {station_name} flood duration: {duration_days} days")

            # Step 5 — Calculate affected area per station
            affected_area = calculate_affected_area(
                station_name  = station_name,
                risk_code     = risk["risk_code"],
                duration_days = duration_days
            )

            # Collect result
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

    # Step 6 — Save all 3 results to database
    if results:
        save_predictions(results)

        # Print summary
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



# ── Test Function With Fake Flood Data ────────────────────
# def test_with_flood_data():
#     """
#     Test pipeline with simulated flood conditions
#     to verify affected area calculations.
#     """
#     print("\n" + "="*50)
#     print("TESTING WITH SIMULATED FLOOD DATA")
#     print("="*50)

#     # Simulate flood conditions
#     test_cases = [
#         {
#             "name": "Test 1 - All Major Flood",
#             "ellagawa":   {"water_level": 13.5, "rainfall": 150.0, "humidity": 98.0, "temperature": 22.0},
#             "putupaula":  {"water_level": 6.0,  "rainfall": 140.0, "humidity": 97.0, "temperature": 22.0},
#             "rathnapura": {"water_level": 11.0, "rainfall": 160.0, "humidity": 99.0, "temperature": 21.0},
#         },
#         {
#             "name": "Test 2 - Only Rathnapura Major",
#             "ellagawa":   {"water_level": 9.0,  "rainfall": 20.0, "humidity": 70.0, "temperature": 26.0},
#             "putupaula":  {"water_level": 2.0,  "rainfall": 15.0, "humidity": 68.0, "temperature": 27.0},
#             "rathnapura": {"water_level": 10.5, "rainfall": 120.0, "humidity": 95.0, "temperature": 23.0},
#         },
#         {
#             "name": "Test 3 - All Normal",
#             "ellagawa":   {"water_level": 8.0, "rainfall": 5.0, "humidity": 65.0, "temperature": 28.0},
#             "putupaula":  {"water_level": 1.5, "rainfall": 3.0, "humidity": 60.0, "temperature": 29.0},
#             "rathnapura": {"water_level": 3.0, "rainfall": 4.0, "humidity": 62.0, "temperature": 28.0},
#         },
#         {
#             "name": "Test 4 - Mixed Levels",
#             "ellagawa":   {"water_level": 12.5, "rainfall": 100.0, "humidity": 92.0, "temperature": 23.0},
#             "putupaula":  {"water_level": 2.0,  "rainfall": 10.0,  "humidity": 65.0, "temperature": 27.0},
#             "rathnapura": {"water_level": 8.0,  "rainfall": 80.0,  "humidity": 90.0, "temperature": 24.0},
#         },
#     ]

#     for test in test_cases:
#         print(f"\n{'─'*40}")
#         print(f"{test['name']}")
#         print(f"{'─'*40}")

#         for station_name in ["Ellagawa", "Putupaula", "Rathnapura"]:
#             data = test[station_name.lower()]

#             # Run flood risk directly with given water level
#             risk = get_flood_risk(
#                 station_name = station_name,
#                 water_level  = data["water_level"],
#                 rainfall     = data["rainfall"],
#                 humidity     = data["humidity"],
#                 temperature  = data["temperature"]
#             )

#             # Calculate affected area
#             affected_area = calculate_affected_area(
#                 station_name  = station_name,
#                 risk_code     = risk["risk_code"],
#                 duration_days = 0.5  # no duration for test
#             )

#             print(f"  {station_name}:")
#             print(f"    Water Level:   {data['water_level']}m")
#             print(f"    Risk Level:    {risk['risk_level']}")
#             print(f"    Affected Area: {affected_area} sq km")


# ── Run directly for testing ───────────────────────────────
if __name__ == "__main__":
    run_pipeline()

# if __name__ == "__main__":
#     # Comment one and uncomment the other to switch modes

#     # ── Real pipeline with Open-Meteo ──
#     # run_pipeline()

#     # ── Test with fake flood data ──
#     test_with_flood_data()