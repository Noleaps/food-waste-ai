import joblib
import pandas as pd
from datetime import datetime

# 1. MEMUAT MODEL DAN FITUR
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # --- LOGIKA WAKTU REAL-TIME ---
    now = datetime.now()
    # Gunakan standar dataset (1-7)
    hari_angka = now.weekday() + 1
    is_weekend = 1 if hari_angka >= 6 else 0

    # 2. MENYIAPKAN INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,  # Suhu standar
        'day_of_week': hari_angka,
        'is_weekend': is_weekend
    }])

    data_input = data_input[features]

    # 3. PREDIKSI DEMAND MURNI
    raw_prediksi = model.predict(data_input)[0]

    # KITA AMBIL ANGKA ASLI AI
    # Kita hanya mencegah angka minus (karena tidak logis)
    # Tapi kita tidak lagi memaksa angka minimal 5% atau 10%
    prediksi_demand = max(0, raw_prediksi)

    # 4. ANALISIS WASTE (SELISIH)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. STRATEGI HARGA & RISIKO
    total_diskon = 0
    risk = "Low Risk"

    # Logika: Jika barang masih lama ( > 14 hari), AI tetap menghitung demand
    # tapi kita menahan diskon demi Profit Margin toko.
    if sisa_hari_expired > 14:
        risk = "Low Risk"
        total_diskon = 0
        catatan = f"Permintaan diprediksi sebanyak {round(prediksi_demand)} unit. Stok aman, pertahankan harga normal."
    else:
        # Penentuan Risiko
        if sisa_hari_expired <= 3 or rasio_waste > 0.6:
            risk = "High Risk"
        elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
            risk = "Medium Risk"

        # Hitung Diskon Berbasis Fuzzy (Bertahap)
        # Diskon hari: 14 hari mulai diskon halus
        diskon_hari = max(0, (14 - sisa_hari_expired) * 4)
        # Diskon stok: semakin banyak yang tidak laku, diskon naik
        diskon_stok = rasio_waste * 20

        total_diskon = round(diskon_hari + diskon_stok)
        catatan = f"Permintaan diprediksi sebanyak {round(prediksi_demand)} unit. Disarankan diskon {total_diskon}% agar stok habis tepat waktu."

    # Batasi diskon maksimal
    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "prediksi_jual_qty": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
