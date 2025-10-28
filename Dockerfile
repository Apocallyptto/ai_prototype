# ===== Base Image =====
FROM python:3.12-slim

# ===== Working Directory =====
WORKDIR /app

# ===== System Dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# ===== Copy App =====
COPY . /app

# ===== Install Python Packages =====
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ===== Environment =====
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ===== Default command (overridden by compose) =====
CMD ["python", "-m", "jobs.executor_bracket"]
