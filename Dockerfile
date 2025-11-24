# Dockerfile — Sovariel 2025 (optimized multi-variant)
FROM python:3.11-slim

# System deps (audio + BLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libsndfile1 libopenblas0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use minimal requirements by default (fast build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY standalone_sim.py .

EXPOSE 5000

# Production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "standalone_sim:app"]
