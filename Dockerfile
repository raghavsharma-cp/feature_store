# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install cron
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY feature_store/ /app/feature_store/
COPY crontab /etc/cron.d/rd_live_cron
COPY entrypoint.sh /entrypoint.sh

# Set permissions
# Note: Cron daemon runs as root, but the cron job will execute as appuser
RUN chmod +x /entrypoint.sh && \
    chmod 0644 /etc/cron.d/rd_live_cron && \
    mkdir -p /app/logs /app/data /var/log && \
    touch /var/log/rd_live.log && \
    chown -R appuser:appuser /app /var/log/rd_live.log

# Keep as root to run cron daemon
# The cron job in /etc/cron.d/ will run as appuser (specified in crontab)

# Run entrypoint script to start cron and keep container running
CMD ["/entrypoint.sh"]