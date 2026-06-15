import joblib
import pandas as pd

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # 1. AI Memprediksi Penjualan
    data_input = pd.DataFrame([{
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,
        'day_of_week': 1,
        'is_weekend': 0,
        'predicted_demand': 50
    }])

    data_input = data_input[features]
    prediksi_terjual = model.predict(data_input)[0]

    # --- LOGIKA FUZZY DISCOUNT (GRADUAL) ---

    # A. Hitung Diskon Berdasarkan Hari (Maksimal 70% di hari ke-1, 0% di hari ke-10)
    if sisa_hari_expired >= 10:
        diskon_hari = 0
    elif sisa_hari_expired <= 0:
        diskon_hari = 80  # Sudah expired / Hari-H
    else:
        # Rumus: Semakin kecil hari, semakin besar diskon secara linear
        diskon_hari = (10 - sisa_hari_expired) * 7.5  # Gradasi per hari 7.5%

    # B. Faktor Pengali Berdasarkan Stok (Waste Risk)
    # Jika stok jauh lebih banyak dari prediksi jual, tambahkan diskon extra
    rasio_sisa = (stok_saat_ini - prediksi_terjual) / (stok_saat_ini + 1)
    # Tambahan maksimal 15% jika barang numpuk
    extra_diskon_stok = max(0, rasio_sisa * 15)

    # C. Total Diskon Akhir
    total_diskon = round(diskon_hari + extra_diskon_stok)

    # Batasi diskon di angka yang masuk akal (5% - 80%)
    if total_diskon < 5:
        total_diskon = 0
    elif total_diskon > 80:
        total_diskon = 80

    # --- PENENTUAN LEVEL RISIKO ---
    if sisa_hari_expired <= 3 or total_diskon >= 50:
        risk_level = "High Risk"
        pesan = f"Risiko tinggi! Stok sisa banyak & expired dekat. Diskon {total_diskon}% segera."
    elif sisa_hari_expired <= 7 or total_diskon >= 20:
        risk_level = "Medium Risk"
        pesan = f"Risiko sedang. Diskon gradual {total_diskon}% untuk mempercepat penjualan."
    else:
        risk_level = "Low Risk"
        pesan = "Kondisi aman. Belum perlu diskon besar."

    return {
        "produk": nama,
        "prediksi_jual_qty": round(float(prediksi_terjual), 2),
        "risiko": risk_level,
        # Hasilnya sekarang dinamis (misal 38%)
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": pesan
    }
