FROM python:3.12-slim

# Workdir inside container
WORKDIR /app

# System deps (for psycopg2, etc.)
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

# Copy project code
COPY . /app

# Default command (overridden by docker-compose services)
CMD ["python", "-m", "services.signal_executor"]
