FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \ 
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh -o install-ollama.sh && \
    bash install-ollama.sh

WORKDIR /label_extractor

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .