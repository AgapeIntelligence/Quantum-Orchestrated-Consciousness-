# Dockerfile — Sovariel 2025 (optimized multi-variant)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (audio, BLAS, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    libopenblas0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY standalone_sim.py .

# Expose port
EXPOSE 5000

# Environment variables for gunicorn
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:5000 --workers=2 --threads=4 --timeout=120 --log-level=info"

# Production-ready server with multi-worker support
CMD ["gunicorn", "standalone_sim:app"]
