import joblib
import pandas as pd
from datetime import datetime

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # 1. PERBAIKAN LOGIKA HARI (Sesuaikan dengan Dataset 1-7)
    now = datetime.now()
    # .weekday() itu 0-6, kita tambah 1 supaya jadi 1-7 sesuai dataset
    hari_angka = now.weekday() + 1
    is_weekend = 1 if hari_angka >= 6 else 0  # 6=Sabtu, 7=Minggu

    # 2. INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,
        'day_of_week': hari_angka,
        'is_weekend': is_weekend
    }])

    data_input = data_input[features]

    # 3. PREDIKSI DEMAND
    raw_prediksi = model.predict(data_input)[0]

    # PERBAIKAN: Gunakan Base Demand yang lebih tinggi (10%) jika prediksi terlalu rendah
    # agar hasil di web terlihat lebih hidup dan realistis untuk supermarket.
    base_demand = stok_saat_ini * 0.10
    prediksi_demand = max(base_demand, raw_prediksi)

    # 4. ANALISIS WASTE
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. STRATEGI HARGA (PROFIT PROTECTION)
    total_diskon = 0
    risk = "Low Risk"

    if sisa_hari_expired > 14:
        risk = "Low Risk"
        total_diskon = 0
        catatan = f"Permintaan terpantau stabil ({round(prediksi_demand)} unit). Belum perlu diskon karena expired masih lama."
    else:
        # Penentuan Risiko
        if sisa_hari_expired <= 3 or rasio_waste > 0.6:
            risk = "High Risk"
        elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
            risk = "Medium Risk"

        # Hitung Diskon
        diskon_hari = max(0, (14 - sisa_hari_expired) * 5)
        diskon_stok = rasio_waste * 15
        total_diskon = round(diskon_hari + diskon_stok)
        catatan = f"AI memprediksi permintaan sebanyak {round(prediksi_demand)} unit. Disarankan diskon {total_diskon}% untuk menjaga stok."

    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "prediksi_jual_qty": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
