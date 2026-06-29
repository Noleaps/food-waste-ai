import joblib
import pandas as pd
from datetime import datetime

# 1. MEMUAT MODEL DAN FITUR
# Pastikan Anda sudah menjalankan train_model.py dan mem-push model.pkl terbaru
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # --- LOGIKA WAKTU REAL-TIME ---
    # Mendapatkan hari saat ini (1=Senin, 7=Minggu) sesuai pola dataset
    now = datetime.now()
    hari_angka = now.weekday() + 1
    is_weekend = 1 if hari_angka >= 6 else 0

    # 2. MENYIAPKAN INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,  # Suhu standar operasional
        'day_of_week': hari_angka,
        'is_weekend': is_weekend
    }])

    # Memastikan urutan kolom sama persis dengan saat training
    data_input = data_input[features]

    # 3. PREDIKSI DEMAND DARI MODEL
    raw_prediksi = model.predict(data_input)[0]

    # --- LOGIKA ANTI-ZERO (SOLUSI UTAMA) ---
    # Karena di dataset Anda angka terkecil adalah 10, kita buat batas bawah 10 unit.
    # Jika stok di bawah 10, kita gunakan 20% dari stok sebagai permintaan minimal.
    demand_minimal = max(10, round(stok_saat_ini * 0.2))
    prediksi_demand = max(demand_minimal, raw_prediksi)

    # 4. ANALISIS WASTE (SELISIH ANTARA STOK DAN RAMALAN PASAR)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. STRATEGI HARGA & RISIKO (PROFIT PROTECTION)
    total_diskon = 0
    risk = "Low Risk"

    # ATURAN A: PERLINDUNGAN PROFIT (> 14 HARI)
    if sisa_hari_expired > 14:
        risk = "Low Risk"
        total_diskon = 0
        catatan = f"Permintaan terpantau stabil ({round(prediksi_demand)} unit). Pertahankan harga normal karena sisa waktu masih {sisa_hari_expired} hari."

    # ATURAN B: EVALUASI DISKON GRADUAL ( <= 14 HARI)
    else:
        # Penentuan Risiko
        if sisa_hari_expired <= 3 or rasio_waste > 0.6:
            risk = "High Risk"
        elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # Hitung Diskon Berbasis Fuzzy (Bertahap/Gradual)
        # Faktor hari: diskon naik setiap hari mendekati kadaluarsa
        diskon_hari = max(0, (14 - sisa_hari_expired) * 4)
        # Faktor stok: diskon naik jika barang numpuk tidak laku
        diskon_stok = rasio_waste * 20

        total_diskon = round(diskon_hari + diskon_stok)
        catatan = f"AI memprediksi permintaan sebanyak {round(prediksi_demand)} unit. Disarankan diskon {total_diskon}% agar stok habis tepat waktu."

    # Batasi diskon maksimal agar tidak rugi total
    total_diskon = min(max(total_diskon, 0), 85)

    # 6. RETURN HASIL (Sinkronisasi dengan UI React)
    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        # Menghilangkan 'undefined'
        "prediksi_jual_qty": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
