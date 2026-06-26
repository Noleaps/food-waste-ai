import pandas as pd
from xgboost import XGBRegressor
import joblib
import os

file_name = 'food_waste_indonesia_dataset_1000.csv'

if not os.path.exists(file_name):
    print(f"Error: File {file_name} tidak ditemukan!")
else:
    # Baca file dengan menghilangkan spasi di nama kolom (agar lebih aman)
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()

    print("Dataset berhasil dibaca!")
    print("Daftar kolom yang ditemukan:", df.columns.tolist())

    # Tentukan Target
    target = 'predicted_demand'

    # Pilih kriteria (Fitur) - Sesuaikan dengan nama yang muncul di daftar tadi
    features = [
        'quantity',  # Sekarang stok (quantity) jadi kriteria/input
        'price_IDR',
        'expiry_days',
        'storage_temperature_C',
        'day_of_week',
        'is_weekend'
    ]
#

    try:
        X = df[features]
        y = df[target]

        print("AI sedang mempelajari data...")
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)

        joblib.dump(model, 'model.pkl')
        joblib.dump(features, 'features.pkl')
        print("-----------------------------------------")
        print("Berhasil! Otak AI Indonesia siap digunakan.")
    except KeyError as e:
        print(f"\nERROR: Kolom {e} tidak ditemukan!")
        print(
            "Pastikan nama kolom di dalam kode 'features' sama dengan daftar kolom di atas.")
