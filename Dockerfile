FROM python:3.10-slim

# System deps for OpenCV headless + librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the 2 MB FER+ ONNX model so containers start instantly
RUN python - <<'EOF'
import urllib.request, os
url  = ("https://github.com/onnx/models/raw/main/validated/"
        "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
dest = "/app/emotion_ferplus.onnx"
print("Downloading FER+ ONNX model…")
urllib.request.urlretrieve(url, dest)
print(f"Done — {os.path.getsize(dest) // 1024} KB")
EOF

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
