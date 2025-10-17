
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install system dependencies (needed for FAISS & scientific libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt


CMD ["uvicorn", "check:app", "--host", "0.0.0.0", "--port", "8080"]
