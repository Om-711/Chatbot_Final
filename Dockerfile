FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set index URL for reliability
RUN pip install --upgrade pip
RUN pip config set global.index-url https://pypi.org/simple

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "check:app", "--host", "0.0.0.0", "--port", "8080"]
