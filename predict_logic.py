import joblib
import pandas as pd

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):

    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,  # Masuk sebagai fitur
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,
        'day_of_week': 1,
        'is_weekend': 0
    }])

    data_input = data_input[features]

    # AI menebak DEMAND (Permintaan)
    prediksi_demand = model.predict(data_input)[0]

    # --- LOGIKA WASTE RISK BARU ---
    # Potensi Waste = Stok - Prediksi Permintaan
    potensi_waste = stok_saat_ini - prediksi_demand

    # Hitung presentase barang yang kemungkinan TIDAK laku
    rasio_waste = potensi_waste / (stok_saat_ini + 1)

    # --- FUZZY DISCOUNT LOGIC ---
    # Dasar diskon dari hari (semakin dekat expired semakin tinggi)
    diskon_hari = max(0, (7 - sisa_hari_expired) *
                      10) if sisa_hari_expired < 7 else 0

    # Bonus diskon dari rasio waste (semakin banyak numpuk, tambah diskon)
    diskon_stok = max(0, rasio_waste * 20)

    total_diskon = round(diskon_hari + diskon_stok)
    total_diskon = min(max(total_diskon, 0), 80)  # Batasi 0-80%

    # --- Tentukan Risiko ---
    if sisa_hari_expired <= 3 or rasio_waste > 0.5:
        risk = "High Risk"
    elif sisa_hari_expired <= 7 or rasio_waste > 0.2:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": f"AI memprediksi permintaan sebanyak {round(prediksi_demand)} unit."
    }
