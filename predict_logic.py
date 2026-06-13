import joblib
import pandas as pd

# Muat model dan fitur yang baru saja Anda buat
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # MENYIAPKAN DATA SESUAI HASIL TRAINING
    # Perhatikan: Nama kolom harus persis 'storage_temperature_C'
    data_input = pd.DataFrame([{
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,  # Suhu rata-rata
        'day_of_week': 1,            # Asumsi hari Senin
        'is_weekend': 0,             # Bukan akhir pekan
        'predicted_demand': 50       # Permintaan rata-rata
    }])

    # Pastikan urutan kolom sesuai dengan features.pkl
    data_input = data_input[features]

    # AI Memprediksi berapa yang akan terjual (quantity)
    prediksi_terjual = model.predict(data_input)[0]

    # --- LOGIKA PENENTUAN RISIKO ---
    potensi_sisa = stok_saat_ini - prediksi_terjual

    risk_level = "Low Risk"
    diskon = 0
    pesan = "Kondisi aman, tidak butuh diskon."

    # Aturan berdasarkan Sisa Hari dan Stok
    if sisa_hari_expired <= 3:
        risk_level = "High Risk"
        diskon = 50
        pesan = "Barang kritis! Segera diskon 50% sebelum kadaluarsa."
    elif sisa_hari_expired <= 7 or potensi_sisa > (0.3 * stok_saat_ini):
        risk_level = "Medium Risk"
        diskon = 20
        pesan = "Stok berisiko menumpuk. Berikan diskon promosi 20%."

    return {
        "produk": nama,
        "prediksi_jual_qty": round(float(prediksi_terjual), 2),
        "risiko": risk_level,
        "diskon_rekomendasi": f"{diskon}%",
        "catatan": pesan
    }
