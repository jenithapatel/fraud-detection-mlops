# Inference API image. Build from repo root: docker build -f docker/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_DATA_DIR=/app/data

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["uvicorn", "fraud.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
