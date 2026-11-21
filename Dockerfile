FROM python:3.11-slim

# Instal library sistem Linux yang dibutuhkan oleh OpenCV & YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instal dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hapus opencv-python versi desktop bawaan ultralytics dan pertahankan versi headless
RUN pip uninstall -y opencv-python && pip install --no-cache-dir --force-reinstall opencv-python-headless

# Salin seluruh file proyek (app.py, best.pt, dll.)
COPY . .

# Jalankan server Gradio
CMD ["python", "app.py"]
