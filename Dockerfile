FROM python:3.12-slim

# Fix for slow / blocked Debian mirrors
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list

# Retry install 3x to avoid transient network failures
RUN for i in 1 2 3; do \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        tzdata ca-certificates curl libpq-dev && \
      update-ca-certificates && \
      break || sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY . /app

ENV TZ=UTC
CMD ["python", "-m", "jobs.cron_v2"]
