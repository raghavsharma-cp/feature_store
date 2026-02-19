# Scripts

## refresh_data

Refreshes PostgreSQL feature store tables from MongoDB: reads all currently admitted patients (same as `feature_store.feature_store`), then replaces rows in `feature_store`, `vitals_feature_store`, `documents_feature_store`, `notes_feature_store`, and `orders_feature_store`.

**Requirements:** `db_uri` (MongoDB) and `postgres_url` or `POSTGRES_URL` (PostgreSQL) in environment or `.env` / `.env.local`.

From project root:

```bash
python -m scripts.refresh_data
# or
PYTHONPATH=. python scripts/refresh_data.py
```

## Create feature store tables

Creates: `feature_store`, `vitals_feature_store`, `documents_feature_store`, `notes_feature_store`, `orders_feature_store`.

With Postgres running (e.g. Docker container named `feature_store`):

```bash
# From project root
docker exec -i feature_store psql -U postgres -d feature_store < scripts/init_feature_store.sql
```

Or with local `psql`:

```bash
psql postgresql://postgres:postgres@localhost:5432/feature_store -f scripts/init_feature_store.sql
```
