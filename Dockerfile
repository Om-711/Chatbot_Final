# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps and tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tini && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY check.py ./
# Optional modules/folders if present
COPY recommendation.py ./ 2>/dev/null || true
COPY templates ./templates 2>/dev/null || true
COPY faiss_index ./faiss_index 2>/dev/null || true

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Runtime envs (values provided at `docker run`)
# - GOOGLE_API_KEY
# - MONGODB_URI
# - ALLOWED_ORIGINS (defaults to permissive via code if unset)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/docs >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "check:app", "--host", "0.0.0.0", "--port", "8000"]


