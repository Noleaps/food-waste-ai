from predict_logic import hitung_rekomendasi

def coba_predict(): 
    nama_produk = "telur"
    harga_idr = 28000
    stok = 50
    sisa_hari_expired = 14

    hasil = hitung_rekomendasi(nama_produk, harga_idr, stok, sisa_hari_expired)

    print("Hasil Prediksi:", hasil)



coba_predict()
