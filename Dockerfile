# Lean image; no apt-get needed
FROM python:3.12-slim

WORKDIR /app

# Copy only reqs first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Now copy the rest of the repo
COPY . /app

# Env & default
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "-m", "jobs.cron_nn"]
