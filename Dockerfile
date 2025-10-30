FROM python:3.12-slim

WORKDIR /app

# system deps for psycopg2 and friends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /app
