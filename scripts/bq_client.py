"""
BigQuery client for feature_store refresh_bq script.

Uses env: BQ_PROJECT_ID, BQ_DATASET_ID, optional BQ_LOCATION.
Credentials: GOOGLE_APPLICATION_CREDENTIALS or BIGQUERY_CREDENTIALS_PATH.
"""

import os
from pathlib import Path
from typing import Optional

from google.cloud import bigquery
from google.oauth2 import service_account


def get_bq_client() -> bigquery.Client:
    """Create a BigQuery client for feature store refresh_bq."""
    project_id = os.environ.get("BQ_PROJECT_ID")
    if not project_id:
        raise ValueError("BQ_PROJECT_ID environment variable is required")
    location = os.environ.get("BQ_LOCATION", "asia-south1")
    credentials_path = os.environ.get("BIGQUERY_CREDENTIALS_PATH") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=project_id, credentials=credentials, location=location)
    return bigquery.Client(project=project_id, location=location)


def get_dataset_id() -> str:
    """Return BQ dataset ID for feature store tables."""
    dataset_id = os.environ.get("BQ_DATASET_ID")
    if not dataset_id:
        raise ValueError("BQ_DATASET_ID environment variable is required")
    return dataset_id


def get_table_ref(table_name: str) -> str:
    """Return fully qualified table ID: project_id.dataset_id.table_name."""
    client = get_bq_client()
    return f"{client.project}.{get_dataset_id()}.{table_name}"
