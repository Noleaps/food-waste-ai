from fastapi import FastAPI
from pydantic import BaseModel
from predict_logic import hitung_rekomendasi
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputProduk(BaseModel):
    nama: str
    harga_idr: float
    stok: int
    sisa_hari_expired: int


@app.post("/predict")
def predict_api(data: InputProduk):
    # Mengirim data ke logika AI yang baru
    hasil = hitung_rekomendasi(
        data.nama,
        data.harga_idr,
        data.sisa_hari_expired,
        data.stok
    )
    return hasil
