FROM python:3.13-slim

WORKDIR /app

# System deps for psycopg2 and timezones
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (compose mounts source too, but this bakes for CI/CD)
COPY . .

ENV PYTHONUNBUFFERED=1

# Default command can be overridden by compose
CMD ["python", "-u", "services/executor.py"]
