"""
Live Respiratory Distress Feature Store
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
from pandas import json_normalize
from typing import Tuple, Optional
from google.cloud import storage
from io import StringIO

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


def extract_hr_rr(feature_store: BaseFeatureStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract HR (Heart Rate) and RR (Respiratory Rate) data from the feature store.
    
    For each row in feature_store.base_df, extracts verified vitals data and creates
    separate rows for HR and RR measurements.
    
    Args:
        feature_store: BaseFeatureStore instance with populated base_df
        
    Returns:
        tuple: (df_hr, df_rr) - Two DataFrames with HR and RR data respectively
    """
    # Schema definitions
    schema_hr = [
        'cpmrn',
        'encounter',
        'hospitalName',
        'unitName',
        'bedNo',
        'HR',
        'vitalTimestamp',
        'admissionTime',
        'isChosenForExperiment',
        'isDischarged'
    ]

    schema_rr = [
        'cpmrn',
        'encounter',
        'hospitalName',
        'unitName',
        'bedNo',
        'RR',
        'vitalTimestamp',
        'admissionTime',
        'isChosenForExperiment',
        'isDischarged'
    ]
    
    # Initialize DataFrames
    df_hr = pd.DataFrame(columns=schema_hr)
    df_rr = pd.DataFrame(columns=schema_rr)
    
    # Lists to store rows before converting to DataFrame
    hr_rows = []
    rr_rows = []
    
    if feature_store.base_df is None or feature_store.base_df.empty:
        logger.warning("feature_store.base_df is empty")
        return df_hr, df_rr
    
    for _, row in feature_store.base_df.iterrows():
        # Get patient identifiers from the row
        cpmrn = row.get('CPMRN')
        encounter = row.get('encounters')
        hospital_name = row.get('hospitalName')
        unit_name = row.get('unitName')
        bed_no = row.get('bedNo')
        admission_time = row.get('ICUAdmitDate')
        
        # Get vitals column
        vitals = row.get('vitals')
        
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
            
            # Create base row data
            base_row_data = {
                'cpmrn': cpmrn,
                'encounter': encounter,
                'hospitalName': hospital_name,
                'unitName': unit_name,
                'bedNo': bed_no,
                'vitalTimestamp': vital_timestamp,
                'admissionTime': admission_time,
                'isChosenForExperiment': False,
                'isDischarged': False
            }
            
            # Add HR row if HR value exists
            if hr_value is not None:
                hr_row = base_row_data.copy()
                hr_row['HR'] = hr_value
                hr_rows.append(hr_row)
            
            # Add RR row if RR value exists
            if rr_value is not None:
                rr_row = base_row_data.copy()
                rr_row['RR'] = rr_value
                rr_rows.append(rr_row)
    
    # Convert lists to DataFrames
    if hr_rows:
        df_hr = pd.DataFrame(hr_rows)
    else:
        df_hr = pd.DataFrame(columns=schema_hr)
    
    if rr_rows:
        df_rr = pd.DataFrame(rr_rows)
    else:
        df_rr = pd.DataFrame(columns=schema_rr)
    
    return df_hr, df_rr

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


def upload_csv_to_gcp(df: pd.DataFrame, bucket_name: str, file_path: str) -> None:
    """
    Upload a DataFrame as CSV to GCP bucket, replacing existing file.
    
    Args:
        df: DataFrame to upload
        bucket_name: Name of the GCP bucket
        file_path: Path to the CSV file in the bucket (e.g., "vitals/HR_VITALS.csv")
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        # Convert DataFrame to CSV string
        csv_content = df.to_csv(index=False)
        
        # Upload to GCP bucket
        blob.upload_from_string(csv_content, content_type='text/csv')
        logger.info(f"Uploaded {file_path} to GCP bucket ({len(df)} rows)")
        
    except Exception as e:
        logger.error(f"Error uploading {file_path} to GCP bucket: {e}", exc_info=True)
        raise


def _merge_vitals_csv(existing_df: pd.DataFrame, new_df: pd.DataFrame, vital_type: str) -> pd.DataFrame:
    """
    Merge new vital data with existing CSV data.
    
    This function performs incremental updates:
    1. Adds new rows for new verified vitals (keeps older records)
    2. Preserves isChosenForExperiment from last record for each cpmrn+encounter
    3. Sets isDischarged=True for all rows if patient is discharged
    4. Defaults isChosenForExperiment to False for new patients
    
    Args:
        existing_df: Existing DataFrame from CSV (may be empty)
        new_df: New DataFrame with vital data
        vital_type: Type of vital ('HR' or 'RR')
    
    Returns:
        Merged DataFrame with all records
    """
    if existing_df.empty and new_df.empty:
        return pd.DataFrame()
    
    # If no existing data, return new data with defaults
    if existing_df.empty:
        if new_df.empty:
            return pd.DataFrame()
        result_df = new_df.copy()
        # Ensure required columns exist
        if 'isChosenForExperiment' not in result_df.columns:
            result_df['isChosenForExperiment'] = False
        if 'isDischarged' not in result_df.columns:
            result_df['isDischarged'] = False
        return result_df
    
    # If no new data, mark all existing patients as discharged
    if new_df.empty:
        result_df = existing_df.copy()
        result_df['isDischarged'] = True
        return result_df
    
    # Ensure required columns exist in both DataFrames
    required_columns = ['cpmrn', 'encounter', 'hospitalName', 'unitName', 'bedNo', 
                        vital_type, 'vitalTimestamp', 'admissionTime', 
                        'isChosenForExperiment', 'isDischarged']
    
    for col in required_columns:
        if col not in existing_df.columns:
            existing_df[col] = False if col in ['isChosenForExperiment', 'isDischarged'] else None
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
        ].copy()
        
        # Get new records for this patient
        new_patient = new_df[
            (new_df['cpmrn'] == cpmrn) & 
            (new_df['encounter'] == encounter)
        ].copy()
        
        # Create set of existing (vitalTimestamp, vital_value) tuples
        existing_keys = set()
        if not existing_patient.empty:
            existing_keys = set(zip(
                existing_patient['vitalTimestamp'],
                existing_patient[vital_type]
            ))
        
        # Get last isChosenForExperiment value from existing records
        last_is_chosen = False
        if not existing_patient.empty:
            last_record = existing_patient.sort_values('vitalTimestamp', ascending=False).iloc[0]
            last_is_chosen = last_record.get('isChosenForExperiment', False)
        
        # Find new records to add
        for _, record in new_patient.iterrows():
            vital_value = record[vital_type]
            vital_timestamp = record['vitalTimestamp']
            
            # Check if this record already exists
            if (vital_timestamp, vital_value) not in existing_keys:
                new_record = {
                    'cpmrn': cpmrn,
                    'encounter': encounter,
                    'hospitalName': record['hospitalName'],
                    'unitName': record['unitName'],
                    'bedNo': record['bedNo'],
                    vital_type: vital_value,
                    'vitalTimestamp': vital_timestamp,
                    'admissionTime': record['admissionTime'],
                    'isChosenForExperiment': last_is_chosen,
                    'isDischarged': False
                }
                new_records.append(new_record)
    
    # Combine existing and new records
    if new_records:
        new_records_df = pd.DataFrame(new_records)
        result_df = pd.concat([existing_df, new_records_df], ignore_index=True)
        logger.info(f"Added {len(new_records)} new {vital_type} records")
    else:
        result_df = existing_df.copy()
    
    return result_df


def update_gcp_csvs(df_hr: pd.DataFrame, df_rr: pd.DataFrame,
                    bucket_name: Optional[str] = None,
                    hr_csv_path: Optional[str] = None,
                    rr_csv_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Update CSV files in GCP bucket with HR and RR vital data.
    
    This function performs incremental updates:
    1. Downloads existing CSVs from GCP bucket
    2. Merges new data with existing data
    3. Uploads updated CSVs back to GCP bucket
    
    Args:
        df_hr: DataFrame with HR vital data
        df_rr: DataFrame with RR vital data
        bucket_name: Name of the GCP bucket (from GCP_BUCKET_NAME env var if not provided)
        hr_csv_path: Path to HR CSV file in bucket (defaults to "HR_VITALS.csv" or with prefix)
        rr_csv_path: Path to RR CSV file in bucket (defaults to "RR_VITALS.csv" or with prefix)
    
    Returns:
        tuple: (hr_rows_added, rr_rows_added) - Number of new rows added for each CSV
    """
    if bucket_name is None:
        bucket_name = os.getenv('GCP_BUCKET_NAME')
        if not bucket_name:
            raise ValueError("GCP_BUCKET_NAME environment variable must be set")
    
    # Get CSV path prefix from environment variable (optional)
    csv_prefix = os.getenv('GCP_CSV_PATH_PREFIX', '').strip()
    if csv_prefix and not csv_prefix.endswith('/'):
        csv_prefix += '/'
    
    # Set default CSV paths
    if hr_csv_path is None:
        hr_csv_path = f"{csv_prefix}HR_VITALS.csv"
    if rr_csv_path is None:
        rr_csv_path = f"{csv_prefix}RR_VITALS.csv"
    
    hr_rows_added = 0
    rr_rows_added = 0
    
    try:
        # Download existing CSVs
        logger.info(f"Downloading existing CSVs from GCP bucket: {bucket_name}")
        existing_hr = download_csv_from_gcp(bucket_name, hr_csv_path)
        existing_rr = download_csv_from_gcp(bucket_name, rr_csv_path)
        
        # Merge HR data
        if not df_hr.empty:
            logger.info(f"Processing HR data ({len(df_hr)} rows)...")
            merged_hr = _merge_vitals_csv(existing_hr, df_hr, 'HR')
            hr_rows_added = len(merged_hr) - len(existing_hr) if not existing_hr.empty else len(merged_hr)
            upload_csv_to_gcp(merged_hr, bucket_name, hr_csv_path)
            logger.info(f"HR CSV updated: {hr_rows_added} new rows added, {len(merged_hr)} total rows")
        elif not existing_hr.empty:
            # No new HR data, but update discharged status
            merged_hr = _merge_vitals_csv(existing_hr, pd.DataFrame(), 'HR')
            upload_csv_to_gcp(merged_hr, bucket_name, hr_csv_path)
        
        # Merge RR data
        if not df_rr.empty:
            logger.info(f"Processing RR data ({len(df_rr)} rows)...")
            merged_rr = _merge_vitals_csv(existing_rr, df_rr, 'RR')
            rr_rows_added = len(merged_rr) - len(existing_rr) if not existing_rr.empty else len(merged_rr)
            upload_csv_to_gcp(merged_rr, bucket_name, rr_csv_path)
            logger.info(f"RR CSV updated: {rr_rows_added} new rows added, {len(merged_rr)} total rows")
        elif not existing_rr.empty:
            # No new RR data, but update discharged status
            merged_rr = _merge_vitals_csv(existing_rr, pd.DataFrame(), 'RR')
            upload_csv_to_gcp(merged_rr, bucket_name, rr_csv_path)
        
        logger.info(f"Successfully updated CSVs in GCP bucket")
        logger.info(f"HR rows added: {hr_rows_added}")
        logger.info(f"RR rows added: {rr_rows_added}")
        
    except Exception as e:
        logger.error(f"Error updating GCP CSVs: {e}", exc_info=True)
        raise
    
    return hr_rows_added, rr_rows_added

def main():
    """
    Main function to load all currently admitted patients, extract HR/RR features,
    and update CSV files in GCP bucket.
    """
    logger.info("Loading Currently Admitted Patients into Feature Store")
    
    # Connect to MongoDB and get all currently admitted patients
    feature_store = BaseFeatureStore()
    patients = feature_store.get_all_currently_admitted_patients()
    
    if not patients:
        logger.warning("No currently admitted patients found.")
        sys.exit(0)
    
    logger.info(f"Processing {len(patients)} currently admitted patients...")
    
    # Process each patient and create base_df
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
    
    # Combine all dataframes
    if all_dfs:
        feature_store.base_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Successfully loaded {len(all_dfs)} patients into base_df")
        logger.info(f"Base DataFrame shape: {feature_store.base_df.shape}")
    else:
        logger.error("No patients were successfully processed.")
        sys.exit(1)
    
    # Extract HR and RR data
    logger.info("Extracting HR and RR Data")
    
    df_hr, df_rr = extract_hr_rr(feature_store)
    
    logger.info(f"HR DataFrame shape: {df_hr.shape}")
    logger.info(f"RR DataFrame shape: {df_rr.shape}")
    
    # Update CSV files in GCP bucket
    logger.info("Updating CSV files in GCP bucket")
    
    update_gcp_csvs(df_hr, df_rr)

    logger.info("Process Complete")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)

