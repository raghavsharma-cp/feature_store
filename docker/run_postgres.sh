#!/usr/bin/env bash
# Run PostgreSQL in Docker with a named volume. Data persists across container restarts.
# Usage: ./run_postgres.sh   (or: sudo ./run_postgres.sh if needed for Docker)

set -e

IMAGE="postgres:16"
CONTAINER_NAME="feature_store"
VOLUME_NAME="feature_store"
DB_NAME="feature_store"
PORT="5432"

# Remove existing container if it exists (volume is preserved)
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d --name "$CONTAINER_NAME" \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB="$DB_NAME" \
  -p "${PORT}:5432" \
  -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
  "$IMAGE"

echo "PostgreSQL container '$CONTAINER_NAME' is running with volume '$VOLUME_NAME'."
echo "Connection: postgresql://postgres:postgres@localhost:${PORT}/${DB_NAME}"
