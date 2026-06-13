import pandas as pd
from xgboost import XGBRegressor
import joblib
import os

# 1. Gunakan file ML
file_name = 'food_waste_ml_1000.csv'

if not os.path.exists(file_name):
    print(f"Error: File {file_name} tidak ditemukan!")
else:
    df = pd.read_csv(file_name)
    print("Dataset berhasil dibaca!")

    # 2. Menentukan Target dan Fitur
    # Target yang mau diprediksi adalah 'quantity'
    target = 'quantity'

    # Kita hanya mengambil kolom yang berisi angka untuk proses belajar
    # Kita buang 'risk_level' dan 'recommended_action' karena itu berbentuk teks
    fitur_angka = df.select_dtypes(include=['number']).columns.tolist()

    if target in fitur_angka:
        fitur_angka.remove(target)  # Buang target dari daftar fitur

    X = df[fitur_angka]
    y = df[target]

    print(f"Menggunakan fitur: {fitur_angka}")
    print(f"Target: {target}")

    # 3. Membuat "Mesin" AI
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)

    # 4. Proses Belajar
    print("AI sedang belajar dari data...")
    model.fit(X, y)

    # 5. Simpan Model dan daftar fitur
    joblib.dump(model, 'model.pkl')
    joblib.dump(fitur_angka, 'features.pkl')

    print("-----------------------------------------")
    print("Berhasil! File 'model.pkl' dan 'features.pkl' telah dibuat.")
    print("Sekarang kita punya 'otak' AI yang siap memprediksi penjualan.")
