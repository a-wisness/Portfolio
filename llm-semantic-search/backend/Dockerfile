# Backend image: FastAPI + sentence-transformers + ChromaDB + Anthropic SDK
FROM python:3.11-slim

# Avoid .pyc files and buffer issues; keep logs flowing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (data/ and .env are intentionally excluded via .dockerignore).
COPY app ./app

EXPOSE 8000

# Persisted at runtime via a named volume mounted at /app/data.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
