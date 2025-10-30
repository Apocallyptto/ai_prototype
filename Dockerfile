# Use 3.12 for wider wheel availability
FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 & timezone/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ca-certificates \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Default command: run your cron loop
CMD ["python", "-m", "jobs.cron_v2"]
