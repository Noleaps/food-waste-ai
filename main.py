from fastapi import FastAPI
from pydantic import BaseModel
from predict_logic import hitung_rekomendasi
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# AGAR UI ANDA BISA MENGAKSES API INI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua alamat mengakses
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisi data apa saja yang harus dikirim dari UI


class DataProduk(BaseModel):
    nama: str
    harga: float
    stok: int
    expired_dalam_hari: int


@app.get("/")
def home():
    return {"status": "AI Food Waste Aktif!"}


@app.post("/predict")
def prediksi_api(item: DataProduk):
    # Memanggil fungsi dari file predict_logic.py
    hasil = hitung_rekomendasi(
        item.nama,
        item.harga,
        item.expired_dalam_hari,
        item.stok
    )
    return hasil
