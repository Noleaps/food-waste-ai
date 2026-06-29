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
    prediksi_demand = max(0, raw_prediksi)

    # 4. ANALISIS WASTE (STOK VS DEMAND)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. STRATEGI HARGA & RISIKO (PENDEKATAN BISNIS REALISTIS)
    total_diskon = 0
    risk = "Low Risk"
    catatan = ""

    # --- ATURAN 1: PERLINDUNGAN MARGIN ( > 14 HARI) ---
    if sisa_hari_expired > 14:
        risk = "Low Risk"
        total_diskon = 0
        catatan = f"Masa kadaluarsa masih lama ({sisa_hari_expired} hari). Pertahankan harga normal untuk menjaga profit margin."

    # --- ATURAN 2: EVALUASI DISKON GRADUAL ( <= 14 HARI) ---
    else:
        # Penentuan Risiko berdasarkan kedekatan hari dan sisa stok
        if sisa_hari_expired <= 3:
            risk = "High Risk"
        elif rasio_waste > 0.5:
            risk = "High Risk"
        elif sisa_hari_expired <= 7 or rasio_waste > 0.2:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # Perhitungan Diskon (Fuzzy)
        # Diskon hanya diberikan jika risiko bukan Low atau stok menumpuk
        if risk != "Low Risk" or rasio_waste > 0.1:
            # Dasar diskon: sisa 14 hari mulai dari 0%, sisa 1 hari menuju 70%
            diskon_hari = max(0, (14 - sisa_hari_expired) * 5)
            # Bonus diskon stok: jika stok banyak yang terancam waste
            diskon_stok = rasio_waste * 15
            total_diskon = round(diskon_hari + diskon_stok)

            catatan = f"Permintaan diprediksi rendah ({round(prediksi_demand)} unit). Diskon {total_diskon}% disarankan karena expired mulai mendekati."
        else:
            total_diskon = 0
            catatan = "Stok dan permintaan seimbang. Pertahankan harga normal."

    # Final touch: Batasi diskon maksimal
    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
