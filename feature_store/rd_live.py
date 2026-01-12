"""
Live Respiratory Distress Feature Store
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from pandas import json_normalize
from google.cloud import storage
import pytz

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up one level to project root
sys.path.insert(0, str(project_root))

from feature_store import BaseFeatureStore, convert_to_serializable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_hr_rr(feature_store: BaseFeatureStore) -> pd.DataFrame:
    """
    Extract HR (Heart Rate) and RR (Respiratory Rate) data from the feature store.
    
    For each patient (cpmrn+encounter), extracts only the latest verified vital that
    has a timestamp within the last 1 hour from the time of execution.
    
    Args:
        feature_store: BaseFeatureStore instance with populated base_df
        
    Returns:
        DataFrame with HR and RR columns, where each row represents the latest vital
        measurement per patient (with both HR and RR values if available) that is
        within the last 1 hour
    """
    schema = [
        'cpmrn',
        'encounter',
        'hospitalName',
        'unitName',
        'bedNo',
        'HR',
        'RR',
        'vitalTimestamp',
        'admissionTime',
        'isChosenForExperiment'
    ]
    
    ist = pytz.timezone('Asia/Kolkata')
    utc = pytz.UTC
    current_time_ist = datetime.now(ist)
    cutoff_time_ist = current_time_ist - timedelta(hours=1)
    
    latest_vitals = {}
    
    if feature_store.base_df is None or feature_store.base_df.empty:
        logger.warning("feature_store.base_df is empty")
        return pd.DataFrame(columns=schema)
    
    for _, row in feature_store.base_df.iterrows():
        cpmrn = row.get('CPMRN')
        encounter = row.get('encounters')
        hospital_name = row.get('hospitalName')
        unit_name = row.get('unitName')
        bed_no = row.get('bedNo')
        admission_time = row.get('ICUAdmitDate')
        vitals = row.get('vitals')
        
        if not isinstance(vitals, list):
            continue
        
        for vital_dict in vitals:
            if not isinstance(vital_dict, dict):
                continue
            
            if not vital_dict.get('isVerified', False):
                continue
            
            vital_timestamp = vital_dict.get('timestamp')
            if vital_timestamp is None:
                continue
            try:
                if isinstance(vital_timestamp, str):
                    vital_dt = pd.to_datetime(vital_timestamp)
                elif isinstance(vital_timestamp, (int, float)):
                    if vital_timestamp > 1e10:
                        vital_dt = datetime.fromtimestamp(vital_timestamp / 1000, tz=utc)
                    else:
                        vital_dt = datetime.fromtimestamp(vital_timestamp, tz=utc)
                elif isinstance(vital_timestamp, datetime):
                    vital_dt = vital_timestamp
                else:
                    vital_dt = pd.to_datetime(vital_timestamp)
                
                if isinstance(vital_dt, pd.Timestamp):
                    if vital_dt.tzinfo is None:
                        vital_dt = vital_dt.tz_localize(utc)
                    else:
                        vital_dt = vital_dt.tz_convert(utc)
                else:
                    if vital_dt.tzinfo is None:
                        vital_dt = utc.localize(vital_dt)
                    else:
                        vital_dt = vital_dt.astimezone(utc)
                
                if isinstance(vital_dt, pd.Timestamp):
                    vital_dt_ist = vital_dt.tz_convert(ist)
                else:
                    vital_dt_ist = vital_dt.astimezone(ist)
                
                if vital_dt_ist < cutoff_time_ist:
                    continue
                    
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"Could not parse vital timestamp {vital_timestamp}: {e}. Skipping vital.")
                continue
            
            hr_value = vital_dict.get('daysHR')
            rr_value = vital_dict.get('daysRR')
            
            if hr_value is not None or rr_value is not None:
                patient_key = (cpmrn, encounter)
                row_data = {
                    'cpmrn': cpmrn,
                    'encounter': encounter,
                    'hospitalName': hospital_name,
                    'unitName': unit_name,
                    'bedNo': bed_no,
                    'HR': hr_value,
                    'RR': rr_value,
                    'vitalTimestamp': vital_timestamp,
                    'admissionTime': admission_time,
                    'isChosenForExperiment': False
                }
                
                if patient_key not in latest_vitals:
                    latest_vitals[patient_key] = (row_data, vital_dt)
                else:
                    existing_dt = latest_vitals[patient_key][1]
                    if vital_dt > existing_dt:
                        latest_vitals[patient_key] = (row_data, vital_dt)
                
    rows = [vital_data[0] for vital_data in latest_vitals.values()]
    
    if rows:
        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(rows)} latest vital records (one per patient) from the last 1 hour")
    else:
        df = pd.DataFrame(columns=schema)
        logger.info("No vital records found in the last 1 hour")
    
    return df


def upload_csv_to_gcp(df: pd.DataFrame, bucket_name: str, file_path: str) -> None:
    """
    Upload a DataFrame as CSV to GCP bucket.
    
    Args:
        df: DataFrame to upload
        bucket_name: Name of the GCP bucket
        file_path: Path to the CSV file in the bucket
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_content = df.to_csv(index=False)
        blob.upload_from_string(csv_content, content_type='text/csv')
        logger.info(f"Uploaded {file_path} to GCP bucket ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error uploading {file_path} to GCP bucket: {e}", exc_info=True)
        raise

def main():
    """
    Load currently admitted patients, extract HR/RR features, and upload to GCP.
    """
    logger.info("Loading Currently Admitted Patients into Feature Store")
    
    feature_store = BaseFeatureStore()
    patients = feature_store.get_all_currently_admitted_patients()
    
    if not patients:
        logger.warning("No currently admitted patients found.")
        sys.exit(0)
    
    logger.info(f"Processing {len(patients)} currently admitted patients...")
    
    all_dfs = []
    for idx, patient in enumerate(patients, 1):
        try:
            serializable_obj = convert_to_serializable(patient)
            df = json_normalize(serializable_obj)
            df = feature_store.getNotesKeys('Diagnosis', df, 'notesDiagnoses')
            df = feature_store.getNotesKeys('Summary', df, 'notesSummary')
            all_dfs.append(df)
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(patients)} patients...")
        except Exception as e:
            logger.error(f"Error processing patient {idx}: {e}", exc_info=True)
            continue
    
    if all_dfs:
        feature_store.base_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Successfully loaded {len(all_dfs)} patients into base_df")
        logger.info(f"Base DataFrame shape: {feature_store.base_df.shape}")
    else:
        logger.error("No patients were successfully processed.")
        sys.exit(1)
    
    logger.info("Extracting HR and RR Data")
    df_vitals = extract_hr_rr(feature_store)

    current_time = datetime.now()
    filename = f"vitals/vitals_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    bucket_name = os.getenv('GCP_BUCKET_NAME')
    if not bucket_name:
        logger.error("GCP_BUCKET_NAME environment variable is not set")
        sys.exit(1)
    
    upload_csv_to_gcp(df_vitals, bucket_name, filename)
    logger.info("Process Complete")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)

