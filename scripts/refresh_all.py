"""
Run a full refresh of Postgres and BigQuery feature store.

1. Refresh Postgres: currently admitted patients from MongoDB (same as scripts.refresh_data).
2. Refresh BQ: patients discharged in the last 2 months from MongoDB (same as scripts.refresh_bq).

Usage (from feature_store project root):
  python -m scripts.refresh_all

Requires: db_uri, postgres_url (or POSTGRES_URL), BQ_PROJECT_ID, BQ_DATASET_ID.
"""

import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_refresh_all() -> None:
    logger.info("Step 1/2: Refreshing Postgres (currently admitted patients from MongoDB)")
    from scripts.refresh_data import run_refresh

    run_refresh()
    logger.info("Step 2/2: Refreshing BigQuery (discharged last 2 months from MongoDB)")
    from scripts.refresh_bq import run_refresh_bq

    run_refresh_bq()
    logger.info("Full refresh complete (Postgres + BQ)")


if __name__ == "__main__":
    run_refresh_all()
