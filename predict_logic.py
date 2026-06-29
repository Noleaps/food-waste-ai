import joblib
import pandas as pd
from datetime import datetime

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')
# Memuat daftar identitas produk
try:
    le = joblib.load('label_encoder.pkl')
except:
    le = None


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    now = datetime.now()
    hari_angka = now.weekday() + 1
    is_weekend = 1 if hari_angka >= 6 else 0

    # Mengubah nama teks ke angka ID yang dipahami AI
    product_id = 0
    if le:
        try:
            # Mencoba mencari ID untuk produk (misal 'Telur' -> 5)
            # Jika produk baru/tidak dikenal, gunakan ID default
            product_id = le.transform([nama])[0]
        except:
            product_id = 0

    # INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'product_encoded': product_id,
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,
        'day_of_week': hari_angka,
        'is_weekend': is_weekend
    }])

    data_input = data_input[features]
    raw_prediksi = model.predict(data_input)[0]

    # --- OUTPUT ASLI (Murni dari otak AI) ---
    # Kita hanya membuang angka minus, tapi TIDAK membatasi ke angka 10 lagi
    prediksi_demand = max(0, raw_prediksi)

    # Logika Risiko & Diskon (Sama seperti sebelumnya)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    risk = "Low Risk"
    total_diskon = 0

    if sisa_hari_expired <= 3 or rasio_waste > 0.6:
        risk = "High Risk"
    elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
        risk = "Medium Risk"

    if sisa_hari_expired <= 14:
        total_diskon = round(
            max(0, (14 - sisa_hari_expired) * 4) + (rasio_waste * 20))

    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "prediksi_jual_qty": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": f"Permintaan diprediksi {round(prediksi_demand)} unit berdasarkan pola belanja hari ini."
    }
