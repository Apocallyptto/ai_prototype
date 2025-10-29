# Use lightweight Python image
FROM python:3.12-slim

# Install system deps for SSL, timezone, and science stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    tzdata \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Default command (can be overridden by compose)
CMD ["python", "-m", "jobs.cron_v2"]
