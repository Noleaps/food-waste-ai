import joblib
import pandas as pd

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama_produk, harga, sisa_hari_expired, stok_sekarang):
    data_input = pd.DataFrame([{
        'price': harga,
        'expiry_days': sisa_hari_expired,
        'temperature': 25,
        'day_of_week': 0,
        'is_weekend': 0,
        'predicted_demand': 0.5,
        'product_encoded': 1
    }])

    data_input = data_input[features]
    prediksi_terjual = model.predict(data_input)[0]

    potensi_sisa = stok_sekarang - prediksi_terjual
    risk_level = "Low Risk"
    diskon = 0
    catatan = "Stok aman."

    if sisa_hari_expired <= 3:
        risk_level = "High Risk"
        diskon = 50
        catatan = "Hampir expired! Berikan diskon besar."
    elif sisa_hari_expired <= 7 or potensi_sisa > 0:
        risk_level = "Medium Risk"
        diskon = 20
        catatan = "Risiko sisa sedang, berikan diskon promosi."

    return {
        "produk": nama_produk,
        "prediksi_terjual": round(float(prediksi_terjual), 2),
        "risiko": risk_level,
        "diskon_rekomendasi": f"{diskon}%",
        "pesan": catatan
    }
