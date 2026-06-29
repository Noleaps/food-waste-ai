import joblib
import pandas as pd
from datetime import datetime

# 1. MEMUAT MODEL DAN FITUR
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # --- LOGIKA HARI DINAMIS ---
    # Mendapatkan hari ini secara otomatis (0=Senin, 1=Selasa, ..., 5=Sabtu, 6=Minggu)
    now = datetime.now()
    hari_angka = now.weekday()

    # AI Studio/Bolt biasanya menyetel tahun 2026 di screenshot Anda,
    # namun kode ini akan mengambil waktu asli komputer server saat ini.

    # Menentukan is_weekend (1 jika Sabtu/Minggu, 0 jika Senin-Jumat)
    is_weekend = 1 if hari_angka >= 5 else 0

    # 2. MENYIAPKAN INPUT UNTUK AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,  # Suhu rata-rata
        'day_of_week': hari_angka,   # DINAMIS mengikuti kalender
        'is_weekend': is_weekend     # DINAMIS mengikuti kalender
    }])

    # Menyesuaikan urutan kolom sesuai features.pkl
    data_input = data_input[features]

    # 3. AI MENEBAK DEMAND (PERMINTAAN)
    raw_prediksi = model.predict(data_input)[0]
    prediksi_demand = max(0, raw_prediksi)  # Pastikan tidak minus

    # 4. ANALISIS WASTE (STOK VS DEMAND)
    # Jika stok 100 tapi demand 120, maka potensi_waste = 0 (barang ludes)
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    # 5. DYNAMIC RISK & FUZZY DISCOUNT
    total_diskon = 0
    risk = "Low Risk"

    # Aturan A: Jika barang ludes terjual (Demand >= Stok), risiko rendah, diskon 0%
    if potensi_waste == 0 and sisa_hari_expired > 3:
        risk = "Low Risk"
        total_diskon = 0
        catatan = f"Permintaan tinggi ({round(prediksi_demand)} unit). Stok akan habis di harga normal."

    # Aturan B: Jika stok berisiko sisa, hitung diskon gradual
    else:
        # Penentuan Risiko
        if sisa_hari_expired <= 3:
            risk = "High Risk"
        elif rasio_waste > 0.6 and sisa_hari_expired <= 10:
            risk = "High Risk"
        elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # Perhitungan Diskon Hanya jika berisiko sisa atau sudah mau expired
        if sisa_hari_expired <= 14 or rasio_waste > 0.1:
            diskon_hari = max(0, (14 - sisa_hari_expired) * 5)
            diskon_stok = rasio_waste * 15
            total_diskon = round(diskon_hari + diskon_stok)

        catatan = f"Permintaan diperkirakan {round(prediksi_demand)} unit. Diskon {total_diskon}% untuk menjaga stok."

    # Final touch: Batasi diskon
    total_diskon = min(max(total_diskon, 0), 85)

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
