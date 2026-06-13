# Menggunakan base image Python
FROM python:3.9-slim

# Menentukan folder kerja di dalam server
WORKDIR /app

# Copy daftar library dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file kode Anda ke dalam server
COPY . .

# Jalankan server FastAPI menggunakan port yang disediakan Google Cloud
CMD uvicorn main:app --host 0.0.0.0 --port $PORT