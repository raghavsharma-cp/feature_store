#!/bin/bash

# Create log file if it doesn't exist and set permissions
touch /var/log/rd_live.log
chown appuser:appuser /var/log/rd_live.log

# Start cron daemon in foreground mode
echo "Starting cron daemon..."
cron

# Tail the log file to keep container running
echo "Cron daemon started. Monitoring logs..."
echo "Cron job will run every hour at minute 0"
tail -f /var/log/rd_live.log

