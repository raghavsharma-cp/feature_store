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
from typing import Optional
from io import StringIO
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

def extract_hr_rr_old(feature_store: BaseFeatureStore) -> pd.DataFrame:
    """
    Extract HR (Heart Rate) and RR (Respiratory Rate) data from the feature store.
    
    For each row in feature_store.base_df, extracts verified vitals data and creates
    rows with both HR and RR columns in the same row.
    
    Args:
        feature_store: BaseFeatureStore instance with populated base_df
        
    Returns:
        DataFrame with HR and RR columns, where each row represents a vital measurement
        with both HR and RR values (if available) at the same timestamp
    """
    # Schema definition with both HR and RR columns
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
        'isChosenForExperiment',
        'isDischarged'
    ]
    
    # List to store rows before converting to DataFrame
    rows = []
    
    if feature_store.base_df is None or feature_store.base_df.empty:
        logger.warning("feature_store.base_df is empty")
        return pd.DataFrame(columns=schema)
    
    for row in feature_store.base_df.itertuples(index=False):
        # Get patient identifiers from the row
        cpmrn = getattr(row, 'CPMRN', None)
        encounter = getattr(row, 'encounters', None)
        hospital_name = getattr(row, 'hospitalName', None)
        unit_name = getattr(row, 'unitName', None)
        bed_no = getattr(row, 'bedNo', None)
        admission_time = getattr(row, 'ICUAdmitDate', None)
        
        # Get vitals column
        vitals = getattr(row, 'vitals', None)
        
        # Skip if vitals is not a list
        if not isinstance(vitals, list):
            continue
        
        # Process each vital dict in the list
        for vital_dict in vitals:
            # Skip if not a dict
            if not isinstance(vital_dict, dict):
                continue
            
            # Skip unverified vitals
            is_verified = vital_dict.get('isVerified', False)
            if not is_verified:
                continue
            
            # Get timestamp from the vital dict
            vital_timestamp = vital_dict.get('timestamp')
            
            # Extract HR value (daysHR)
            hr_value = vital_dict.get('daysHR')
            
            # Extract RR value (daysRR)
            rr_value = vital_dict.get('daysRR')
            
            # Only create a row if at least one value exists
            if hr_value is not None or rr_value is not None:
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
                    'isChosenForExperiment': False,
                    'isDischarged': False
                }
                rows.append(row_data)
    
    # Convert list to DataFrame
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=schema)
    
    return df

def download_csv_from_gcp(bucket_name: str, file_path: str) -> Optional[pd.DataFrame]:
    """
    Download a CSV file from GCP bucket and return as DataFrame.
    
    Args:
        bucket_name: Name of the GCP bucket
        file_path: Path to the CSV file in the bucket (e.g., "vitals/HR_VITALS.csv")
    
    Returns:
        DataFrame with the CSV data, or empty DataFrame if file doesn't exist
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            logger.info(f"CSV file {file_path} does not exist in bucket {bucket_name}, starting fresh")
            return pd.DataFrame()
        
        # Download CSV content as string
        csv_content = blob.download_as_text()
        
        # Read into DataFrame
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"Downloaded {file_path} from GCP bucket ({len(df)} rows)")
        return df
        
    except Exception as e:
        logger.warning(f"Error downloading {file_path} from GCP bucket: {e}. Starting fresh.")
        return pd.DataFrame()


def _merge_vitals_csv(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new vital data with existing CSV data.
    
    This function performs incremental updates:
    1. Adds new rows for new verified vitals (keeps older records)
    2. Preserves isChosenForExperiment from last record for each cpmrn+encounter
    3. Sets isDischarged=True for all rows if patient is discharged
    4. Defaults isChosenForExperiment to False for new patients
    
    Args:
        existing_df: Existing DataFrame from CSV (may be empty)
        new_df: New DataFrame with vital data (contains both HR and RR columns)
    
    Returns:
        Merged DataFrame with all records
    """
    if existing_df.empty and new_df.empty:
        return pd.DataFrame()
    
    # If no existing data, return new data with defaults
    if existing_df.empty:
        if new_df.empty:
            return pd.DataFrame()
        result_df = new_df
        # Ensure required columns exist
        if 'isChosenForExperiment' not in result_df.columns:
            result_df['isChosenForExperiment'] = False
        if 'isDischarged' not in result_df.columns:
            result_df['isDischarged'] = False
        if 'HR' not in result_df.columns:
            result_df['HR'] = None
        if 'RR' not in result_df.columns:
            result_df['RR'] = None
        return result_df
    
    # If no new data, mark all existing patients as discharged
    if new_df.empty:
        existing_df['isDischarged'] = True
        return existing_df
    
    # Ensure required columns exist in both DataFrames
    required_columns = ['cpmrn', 'encounter', 'hospitalName', 'unitName', 'bedNo', 
                        'HR', 'RR', 'vitalTimestamp', 'admissionTime', 
                        'isChosenForExperiment', 'isDischarged']
    
    for col in required_columns:
        if col not in existing_df.columns:
            if col in ['isChosenForExperiment', 'isDischarged']:
                existing_df[col] = False
            else:
                existing_df[col] = None
        if col not in new_df.columns:
            if col == 'isChosenForExperiment':
                new_df[col] = False
            elif col == 'isDischarged':
                new_df[col] = False
            else:
                new_df[col] = None
    
    # Get unique cpmrn+encounter combinations from existing data
    existing_combinations = set(zip(existing_df['cpmrn'], existing_df['encounter']))
    
    # Get unique cpmrn+encounter combinations from new data
    new_combinations = set(zip(new_df['cpmrn'], new_df['encounter']))
    
    # Mark discharged patients (in existing but not in new)
    discharged_combinations = existing_combinations - new_combinations
    for cpmrn, encounter in discharged_combinations:
        mask = (existing_df['cpmrn'] == cpmrn) & (existing_df['encounter'] == encounter)
        existing_df.loc[mask, 'isDischarged'] = True
    
    # Process each patient in new data
    new_records = []
    
    for cpmrn, encounter in new_combinations:
        # Get existing records for this patient
        existing_patient = existing_df[
            (existing_df['cpmrn'] == cpmrn) & 
            (existing_df['encounter'] == encounter)
        ]
        
        # Get new records for this patient
        new_patient = new_df[
            (new_df['cpmrn'] == cpmrn) & 
            (new_df['encounter'] == encounter)
        ]
        
        # Create set of existing (vitalTimestamp, HR, RR) tuples for deduplication
        existing_keys = set()
        if not existing_patient.empty:
            existing_keys = set(zip(
                existing_patient['vitalTimestamp'],
                existing_patient['HR'].fillna('').astype(str),
                existing_patient['RR'].fillna('').astype(str)
            ))
        
        # Get last isChosenForExperiment value from existing records
        last_is_chosen = False
        if not existing_patient.empty:
            last_record = existing_patient.sort_values('vitalTimestamp', ascending=False).iloc[0]
            last_is_chosen = last_record.get('isChosenForExperiment', False)
        
        # Find new records to add
        for record in new_patient.itertuples(index=False):
            hr_value = getattr(record, 'HR', None)
            rr_value = getattr(record, 'RR', None)
            vital_timestamp = getattr(record, 'vitalTimestamp', None)
            
            # Check if this record already exists (same timestamp and same HR/RR values)
            hr_str = str(hr_value) if pd.notna(hr_value) else ''
            rr_str = str(rr_value) if pd.notna(rr_value) else ''
            
            if (vital_timestamp, hr_str, rr_str) not in existing_keys:
                new_record = {
                    'cpmrn': cpmrn,
                    'encounter': encounter,
                    'hospitalName': getattr(record, 'hospitalName', None),
                    'unitName': getattr(record, 'unitName', None),
                    'bedNo': getattr(record, 'bedNo', None),
                    'HR': hr_value,
                    'RR': rr_value,
                    'vitalTimestamp': vital_timestamp,
                    'admissionTime': getattr(record, 'admissionTime', None),
                    'isChosenForExperiment': last_is_chosen,
                    'isDischarged': False
                }
                new_records.append(new_record)
    
    # Combine existing and new records
    if new_records:
        new_records_df = pd.DataFrame(new_records)
        result_df = pd.concat([existing_df, new_records_df], ignore_index=True)
        logger.info(f"Added {len(new_records)} new vital records")
        del new_records_df
    else:
        result_df = existing_df
    
    return result_df


def update_gcp_csvs(df_vitals: pd.DataFrame,
                    bucket_name: Optional[str] = None,
                    csv_path: Optional[str] = None) -> int:
    """
    Update CSV file in GCP bucket with HR and RR vital data.
    
    This function performs incremental updates:
    1. Downloads existing CSV from GCP bucket
    2. Merges new data with existing data
    3. Uploads updated CSV back to GCP bucket
    
    Args:
        df_vitals: DataFrame with HR and RR vital data (both columns in same DataFrame)
        bucket_name: Name of the GCP bucket (from GCP_BUCKET_NAME env var if not provided)
        csv_path: Path to CSV file in bucket (defaults to "VITALS.csv" or with prefix)
    
    Returns:
        Number of new rows added
    """
    if bucket_name is None:
        bucket_name = os.getenv('GCP_BUCKET_NAME')
        if not bucket_name:
            raise ValueError("GCP_BUCKET_NAME environment variable must be set")
    
    # Get CSV path prefix from environment variable (optional)
    csv_prefix = os.getenv('GCP_CSV_PATH_PREFIX', '').strip()
    if csv_prefix and not csv_prefix.endswith('/'):
        csv_prefix += '/'
    
    # Set default CSV path
    if csv_path is None:
        csv_path = f"{csv_prefix}VITALS.csv"
    
    rows_added = 0
    
    try:
        # Download existing CSV
        logger.info(f"Downloading existing CSV from GCP bucket: {bucket_name}")
        existing_df = download_csv_from_gcp(bucket_name, csv_path)
        
        # Merge data
        if not df_vitals.empty:
            logger.info(f"Processing vital data ({len(df_vitals)} rows)...")
            merged_df = _merge_vitals_csv(existing_df, df_vitals)
            rows_added = len(merged_df) - len(existing_df) if not existing_df.empty else len(merged_df)
            upload_csv_to_gcp(merged_df, bucket_name, csv_path)
            logger.info(f"CSV updated: {rows_added} new rows added, {len(merged_df)} total rows")
            del merged_df
        elif not existing_df.empty:
            # No new data, but update discharged status
            merged_df = _merge_vitals_csv(existing_df, pd.DataFrame())
            upload_csv_to_gcp(merged_df, bucket_name, csv_path)
            del merged_df
        
        logger.info(f"Successfully updated CSV in GCP bucket")
        logger.info(f"Rows added: {rows_added}")
        
    except Exception as e:
        logger.error(f"Error updating GCP CSV: {e}", exc_info=True)
        raise
    
    return rows_added


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
    
    for row in feature_store.base_df.itertuples(index=False):
        cpmrn = getattr(row, 'CPMRN', None)
        encounter = getattr(row, 'encounters', None)
        hospital_name = getattr(row, 'hospitalName', None)
        unit_name = getattr(row, 'unitName', None)
        bed_no = getattr(row, 'bedNo', None)
        admission_time = getattr(row, 'ICUAdmitDate', None)
        vitals = getattr(row, 'vitals', None)
        
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
    
    feature_store.base_df = pd.DataFrame()
    processed_count = 0
    
    for idx, patient in enumerate(patients, 1):
        try:
            serializable_obj = convert_to_serializable(patient)
            df = json_normalize(serializable_obj)
            df = feature_store.getNotesKeys('Diagnosis', df, 'notesDiagnoses')
            df = feature_store.getNotesKeys('Summary', df, 'notesSummary')
            
            if feature_store.base_df.empty:
                feature_store.base_df = df
            else:
                feature_store.base_df = pd.concat([feature_store.base_df, df], ignore_index=True)
            
            processed_count += 1
            del df, serializable_obj
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(patients)} patients...")
        except Exception as e:
            logger.error(f"Error processing patient {idx}: {e}", exc_info=True)
            continue
    
    if processed_count > 0:
        logger.info(f"Successfully loaded {processed_count} patients into base_df")
        logger.info(f"Base DataFrame shape: {feature_store.base_df.shape}")
    else:
        logger.error("No patients were successfully processed.")
        sys.exit(1)
    
    del patients
    
    logger.info("Extracting HR and RR Data")
    df_vitals_old = extract_hr_rr_old(feature_store)
    logger.info(f"Old Vitals DataFrame shape: {df_vitals_old.shape}")
    
    # Update CSV file in GCP bucket
    logger.info("Updating Old Vitals CSV file in GCP bucket")
    
    update_gcp_csvs(df_vitals_old)
    del df_vitals_old

    df_vitals = extract_hr_rr(feature_store)
    logger.info(f"New Vitals DataFrame shape: {df_vitals.shape}")

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

