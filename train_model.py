import pandas as pd
from xgboost import XGBRegressor
import joblib
import os
from sklearn.preprocessing import LabelEncoder

file_name = 'food_waste_indonesia_dataset_1000.csv'

if not os.path.exists(file_name):
    print("Error: File tidak ditemukan!")
else:
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()

    # Ubah Nama Produk Jadi Angka
    le = LabelEncoder()
    df['product_encoded'] = le.fit_transform(df['product'])
    # Simpan daftar nama produk agar AI tahu ID 1 itu apa
    joblib.dump(le, 'label_encoder.pkl')

    target = 'predicted_demand'
    features = [
        'product_encoded',
        'quantity',
        'price_IDR',
        'expiry_days',
        'storage_temperature_C',
        'day_of_week',
        'is_weekend'
    ]

    X = df[features]
    y = df[target]

    print("AI sedang belajar mengenali jenis-jenis produk...")
    # Tuning agar lebih sensitif
    model = XGBRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'features.pkl')
    print("Berhasil! AI sekarang sudah kenal perbedaan tiap produk.")
