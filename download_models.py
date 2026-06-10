import gdown
import os

os.makedirs('models', exist_ok=True)

files = {
    'models/flood_risk_model.h5':                '1rNO73rEqaHR_Cubg9KkMamWi6J7MuadI',
    'models/floodsense_lstm_scaler.pkl':          '1Aty83aYKuitnO26Fb0liEK-xU3Ls7UCE',
    'models/water_level_xgb_model.joblib':        '1sP8dSeLpBaEVE0G__aPuGTYcDXQG2uDD',
    'models/water_level_xgb_features.joblib':     '1L-qj2IxDTBsX71UOK0ziWvq1WbNb1m9P',
    'models/water_level_xgb_station_map.joblib':  '1totuyb4Hd7pjL7Fs6AHJL-vlfEy3wxFT',
}

for path, file_id in files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            path,
            quiet=False
        )
        print(f"✅ {path} downloaded!")
    else:
        print(f"✅ {path} already exists, skipping.")

print("\n✅ All models ready!")

# done changes