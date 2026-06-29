import joblib
import pandas as pd

# Memuat model dan daftar fitur pendukung
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


def hitung_rekomendasi(nama, harga_idr, sisa_hari_expired, stok_saat_ini):
    # 1. Menyiapkan data untuk dikirim ke AI
    data_input = pd.DataFrame([{
        'quantity': stok_saat_ini,
        'price_IDR': harga_idr,
        'expiry_days': sisa_hari_expired,
        'storage_temperature_C': 24,  # Suhu standar
        'day_of_week': 1,            # Asumsi Senin
        'is_weekend': 0
    }])

    # Menyamakan urutan kolom sesuai keinginan AI
    data_input = data_input[features]

    # 2. AI menebak DEMAND (Permintaan Pembeli)
    raw_prediksi = model.predict(data_input)[0]

    # PERBAIKAN: Memastikan angka permintaan tidak minus (Minimal 0)
    prediksi_demand = max(0, raw_prediksi)

    # 3. MENGHITUNG RISIKO (Logic yang lebih cerdas)
    # Menghitung selisih stok dengan permintaan
    potensi_waste = max(0, stok_saat_ini - prediksi_demand)
    rasio_waste = potensi_waste / (stok_saat_ini + 0.1)

    risk = "Low Risk"
    # Barang High Risk jika: sudah mau expired (3 hari)
    # ATAU stok sangat menumpuk tapi expired sudah mulai dekat (dibawah 10 hari)
    if sisa_hari_expired <= 3:
        risk = "High Risk"
    elif rasio_waste > 0.6 and sisa_hari_expired <= 10:
        risk = "High Risk"
    elif sisa_hari_expired <= 7 or rasio_waste > 0.3:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # 4. LOGIKA DISKON FUZZY (Bertahap)
    total_diskon = 0
    if risk != "Low Risk":
        # Diskon bertambah seiring berkurangnya hari (Maksimal di hari ke-0)
        # Kami menggunakan rentang 14 hari untuk mulai diskon halus
        diskon_hari = max(0, (14 - sisa_hari_expired) * 5)

        # Tambahan diskon jika rasio barang sisa banyak
        diskon_stok = rasio_waste * 15

        total_diskon = round(diskon_hari + diskon_stok)

    # Batasi diskon maksimal 85% dan minimal 0%
    total_diskon = min(max(total_diskon, 0), 85)

    # 5. MENYUSUN PESAN REKOMENDASI
    if risk == "Low Risk":
        catatan = f"Permintaan stabil. AI memprediksi {round(prediksi_demand)} unit akan terjual."
    else:
        catatan = f"Permintaan rendah ({round(prediksi_demand)} unit). Disarankan diskon untuk menghabiskan sisa {round(potensi_waste)} unit."

    return {
        "produk": nama,
        "prediksi_demand": round(float(prediksi_demand), 2),
        "risiko": risk,
        "diskon_rekomendasi": f"{total_diskon}%",
        "catatan": catatan
    }
