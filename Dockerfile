FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash bot \
    && mkdir -p /app/data /app/logs \
    && chown -R bot:bot /app

USER bot

# Default to a Day-1-safe boot. Override via `docker run ... --mode PAPER`.
CMD ["python", "-m", "app.main", "--mode", "PAPER", "--dry-run"]
