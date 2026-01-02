#!/bin/bash

# Entrypoint script for rd_live cron job container

set -e

echo "Starting rd_live cron job container..."

# Load environment variables if .env.local exists
if [ -f /app/.env.local ]; then
    echo "Loading environment variables from .env.local..."
    export $(cat /app/.env.local | grep -v '^#' | xargs)
fi

# Ensure log file exists and is writable
touch /var/log/rd_live.log
chmod 666 /var/log/rd_live.log

# Print environment info (without sensitive data)
echo "Environment variables loaded"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Start cron in foreground
echo "Starting cron daemon..."
exec "$@"

