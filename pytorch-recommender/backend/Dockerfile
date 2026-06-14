# Backend image: FastAPI + CPU PyTorch NeuMF recommender.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first for better layer caching (CPU torch wheel via requirements).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the standalone scripts.
COPY app ./app
COPY scripts ./scripts

# Bake the committed checkpoint into the image as a seed. The entrypoint copies
# it into the (initially empty) artifacts volume on first boot, so a plain
# `docker compose up` serves immediately; retraining writes to the volume and
# persists.
COPY artifacts ./seed_artifacts
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
