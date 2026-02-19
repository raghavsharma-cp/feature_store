#!/bin/bash

# Save environment variables to a file that cron can access
# Cron doesn't inherit env vars, so we write them to a file
mkdir -p /etc/environment.d
echo "export db_uri='${db_uri}'" > /etc/environment.d/rd_live.env
echo "export CSV_OUTPUT_PATH='${CSV_OUTPUT_PATH}'" >> /etc/environment.d/rd_live.env
chmod 644 /etc/environment.d/rd_live.env
chown appuser:appuser /etc/environment.d/rd_live.env

# Create a wrapper script that sources the environment file
cat > /usr/local/bin/run_rd_live.sh << 'EOF'
#!/bin/bash
# Source environment variables from file
if [ -f /etc/environment.d/rd_live.env ]; then
    source /etc/environment.d/rd_live.env
fi
cd /app && /opt/venv/bin/python -m feature_store.rd_live
EOF

chmod +x /usr/local/bin/run_rd_live.sh
chown appuser:appuser /usr/local/bin/run_rd_live.sh

# Create log file if it doesn't exist and set permissions
touch /var/log/rd_live.log
chown appuser:appuser /var/log/rd_live.log

# Update crontab to use the wrapper script
echo "0 * * * * appuser /usr/local/bin/run_rd_live.sh >> /var/log/rd_live.log 2>&1" > /etc/cron.d/rd_live_cron
chmod 0644 /etc/cron.d/rd_live_cron

# Clean up any stale cron processes and lock files before starting
echo "Cleaning up any stale cron processes..."
pkill -9 cron 2>/dev/null || true
rm -f /var/run/crond.pid
sleep 1

# Start cron daemon in foreground mode
echo "Starting cron daemon..."
cron

# Tail the log file to keep container running
echo "Cron daemon started. Monitoring logs..."
echo "Cron job will run every hour at minute 0"
tail -f /var/log/rd_live.log

