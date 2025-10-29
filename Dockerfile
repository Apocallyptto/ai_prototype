# executor/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

WORKDIR /app

# Optional: if your base image uses debian.sources, flip to https safely (ignore if file’s absent)
# RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources || true

# System deps (minimal): build tools for some wheels, libpq-dev for psycopg2, CA certs & tzdata
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        tzdata \
        ca-certificates \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (for layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# App code
COPY . /app

# Default command is your cron_v2 loop (overridden by docker compose if you exec into the container)
CMD ["python", "-m", "jobs.cron_v2"]
