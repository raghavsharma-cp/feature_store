# Feature Store

This project provides a feature store for processing patient data from MongoDB and extracting respiratory distress features.

## Project Structure

```
feature_store/
├── feature_store/          # Main package
│   ├── __init__.py
│   ├── feature_store.py     # Base feature store class
│   ├── rd_live.py          # Live respiratory distress feature extraction
│   └── resp_distress_features.py
├── docker/                 # Docker configuration files
│   └── rd_live/
│       ├── Dockerfile      # Docker image definition
│       ├── docker-compose.yml  # Docker Compose configuration
│       ├── crontab         # Cron schedule configuration
│       └── entrypoint.sh  # Container entrypoint script
├── requirements.txt       # Python dependencies
└── .env.local            # Environment variables (not in git)

```

## Setup

### 0. GPU Device Setup (Optional)

If you're running this on a GPU-enabled device (for downstream access to outputs):

1. **Install NVIDIA Drivers** (if not already installed):
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-<version>
   # Reboot if needed
   ```

2. **Install NVIDIA Container Toolkit**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify GPU Access**:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

**Note**: The cron job itself does not use GPU acceleration. GPU configuration is included only to allow the container to run on GPU-enabled devices for easier access to outputs by downstream processes.

### 1. Configure Environment Variables

Create `.env.local` in the project root with:

```bash
# MongoDB connection (required)
db_uri=mongodb://your_mongodb_connection_string

# GCP Bucket Configuration (for storing CSV files)
GCP_BUCKET_NAME=your-gcp-bucket-name
# Optional: Path prefix for CSV files in bucket (defaults to root)
# GCP_CSV_PATH_PREFIX=vitals

# GCP Authentication (choose one):
# Option 1: Service Account JSON key file path
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-credentials.json
# Option 2: Use default credentials (if running on GCP)
# Leave GOOGLE_APPLICATION_CREDENTIALS unset
```

**Note**: The system connects directly to MongoDB using the `db_uri` connection string to query currently admitted patients. The HR and RR vitals data is stored as CSV files in the GCP bucket, with the same schema as the previous PostgreSQL tables.

### 2. Update Cron Schedule (Optional)

Edit `docker/rd_live/crontab` to change the schedule. Current schedule runs every 6 hours at minute 0 (00:00, 06:00, 12:00, 18:00):

```bash
# Format: minute hour day month weekday command
0 */6 * * * cd /app && /usr/local/bin/python -m feature_store.rd_live >> /var/log/rd_live.log 2>&1
```

Common schedules:
- Every hour: `0 * * * *`
- Every 6 hours: `0 */6 * * *` (current)
- Every 30 minutes: `*/30 * * * *`
- Every 15 minutes: `*/15 * * * *`
- Daily at 2 AM: `0 2 * * *`

## Usage

### Using Docker Compose (Recommended)

```bash
# Navigate to docker directory
cd docker/rd_live

# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker directly

```bash
# Build the image (from project root)
docker build -f docker/rd_live/Dockerfile -t rd_live_cron .

# Run the container
docker run -d \
  --name rd_live_cron \
  --restart unless-stopped \
  -v $(pwd)/.env.local:/app/.env.local:ro \
  rd_live_cron

# View logs
docker logs -f rd_live_cron

# View cron logs
docker exec rd_live_cron tail -f /var/log/rd_live.log
```

## Monitoring

### View Application Logs

```bash
# Docker Compose
docker-compose logs -f rd_live_cron

# Docker
docker logs -f rd_live_cron
```

### View Cron Execution Logs

```bash
# Docker Compose
docker-compose exec rd_live_cron tail -f /var/log/rd_live.log

# Docker
docker exec rd_live_cron tail -f /var/log/rd_live.log
```

### Check Cron Status

```bash
docker exec rd_live_cron crontab -l
```

### Test the Script Manually

```bash
docker exec rd_live_cron python -m feature_store.rd_live
```

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs rd_live_cron
```

### Cron job not running

1. Check if cron is running:
```bash
docker exec rd_live_cron pgrep cron
```

2. Check cron logs:
```bash
docker exec rd_live_cron tail -f /var/log/rd_live.log
```

3. Test the script manually:
```bash
docker exec rd_live_cron python -m feature_store.rd_live
```

### Environment variables not loading

Ensure `.env.local` is mounted correctly:
```bash
docker exec rd_live_cron cat /app/.env.local
```

## Updating the Schedule

1. Edit `docker/rd_live/crontab`
2. Rebuild the container:
```bash
cd docker/rd_live
docker-compose build
docker-compose up -d
```

Or restart the container:
```bash
cd docker/rd_live
docker-compose restart
```

## Local Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Script Locally

```bash
python -m feature_store.rd_live
```

Make sure `.env.local` exists in the project root with the required environment variables.

## GPU Device Support

The Docker setup includes GPU device access configuration to allow the container to run on GPU-enabled devices. This enables downstream processes to easily access the cron job outputs on the same device.

**Important**: The cron job itself does not use GPU acceleration - it runs entirely on CPU. GPU configuration is included only for device compatibility.

### Verifying GPU Access (Optional)

If running on a GPU-enabled device, you can verify GPU is accessible:

```bash
# Check GPU inside the container
docker exec rd_live_cron nvidia-smi
```

### GPU Device Requirements (Optional)

If running on a GPU-enabled device:
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on host
- NVIDIA Container Toolkit installed
- Docker with GPU runtime support

The cron job will run normally whether GPU is available or not - it does not require GPU to function.

