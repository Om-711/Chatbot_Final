FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install system dependencies needed for FAISS & scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port 8080 (Render requires binding to $PORT environment variable)
ENV PORT=8080

# Run the FastAPI app
CMD ["uvicorn", "check:app", "--host", "0.0.0.0", "--port", "8080"]

