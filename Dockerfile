# Dockerfile — Python 3.11, tuned for Render + common system deps
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system packages needed by some python packages (faiss, sentence-transformers)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app sources
COPY . /app

# Make sure render's $PORT will be used by the start command (Render sets $PORT env).
EXPOSE 8000

# Use uvicorn (Render will set $PORT). Use single worker to save memory.
CMD ["sh", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --loop uvloop --ws none"]
