FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libsndfile1 libopenblas0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY standalone_sim.py .

EXPOSE 5000
CMD ["python", "standalone_sim.py"]
