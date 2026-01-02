# Dockerfile for rd_live.py cron job
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy feature_store package
COPY feature_store/ /app/feature_store/

# Create directory for logs and data
RUN mkdir -p /app/logs /app/data

# Copy cron configuration
COPY docker/rd_live/crontab /etc/cron.d/rd-live-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/rd-live-cron

# Apply cron job
RUN crontab /etc/cron.d/rd-live-cron

# Copy entrypoint script
COPY docker/rd_live/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create log file with proper permissions
RUN touch /var/log/rd_live.log && chmod 666 /var/log/rd_live.log

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Keep container running
CMD ["cron", "-f"]

