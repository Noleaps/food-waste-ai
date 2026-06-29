import joblib
import pandas as pd
from datetime import datetime

# 1. MEMUAT MODEL DAN FITUR
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # --- LOGIKA HARI DINAMIS ---
    now = datetime.now()
    hari_angka = now.weekday()
    is_weekend = 1 if hari_angka >= 5 else 0

    # 2. MENYIAPKAN INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,
        'day_of_week': hari_angka,
        'is_weekend': is_weekend
    }])

    data_input = data_input[features]

    # 3. AI MENEBAK DEMAND (PERMINTAAN)
    raw_prediksi = model.predict(data_input)[0]

    # SOLUSI AGAR TIDAK 0: Berikan Base Demand (Permintaan Dasar)
    # minimal 5% dari jumlah stok agar perhitungan tetap berjalan secara logis.
    base_demand = stok_saat_ini * 0.05
    prediksi_demand = max(base_demand, raw_prediksi)

    # 4. ANALISIS WASTE (STOK VS DEMAND)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. STRATEGI HARGA & RISIKO
    total_diskon = 0
    risk = "Low Risk"
    catatan = ""

    # --- PENENTUAN RISIKO ---
    if sisa_hari_expired <= 3:
        risk = "High Risk"
    elif rasio_waste > 0.6 and sisa_hari_expired <= 10:
        risk = "High Risk"
    elif sisa_hari_expired <= 14 or rasio_waste > 0.3:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # --- PERHITUNGAN DISKON (FUZZY) ---
    # Kita mulai hitung diskon dari sisa 21 hari agar lebih halus/gradual
    if sisa_hari_expired <= 21:
        # Faktor hari: Makin dekat makin tinggi (skala halus)
        diskon_hari = max(0, (21 - sisa_hari_expired) * 2.5)
        # Faktor stok: Makin banyak sisa makin tinggi
        diskon_stok = rasio_waste * 10
        total_diskon = round(diskon_hari + diskon_stok)
    else:
        total_diskon = 0

    # Koreksi catatan agar informatif
    if total_diskon > 0:
        catatan = f"Permintaan diprediksi {round(prediksi_demand)} unit. Disarankan diskon {total_diskon}% untuk mengantisipasi sisa stok."
    else:
        catatan = f"Masa kadaluarsa masih lama ({sisa_hari_expired} hari) dan permintaan stabil."

    # Final touch: Batasi diskon maksimal 85%
    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        # FIX: Menghapus 'undefined' di UI
        "prediksi_jual_qty": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
