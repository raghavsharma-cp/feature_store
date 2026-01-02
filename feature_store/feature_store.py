import os
import logging
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from pandas import json_normalize
import pandas as pd
from dotenv import load_dotenv
from typing import List, Any, Optional

load_dotenv('.env.local')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) 


def convert_to_serializable(obj):
    """
    Recursively convert MongoDB ObjectId and datetime objects to strings
    to make them JSON serializable for DataFrame conversion.
    
    Args:
        obj: Object to convert (can be dict, list, ObjectId, datetime, or primitive)
        
    Returns:
        Converted object with ObjectId and datetime as strings
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


class BaseFeatureStore:
    """
    Base class for feature store operations.
    
    This class handles fetching patient data from MongoDB and creating base DataFrames.
    Instances can be created in multiple files to add specific features.
    """
    
    def __init__(self, base_df: Optional[pd.DataFrame] = None):
        """
        Initialize BaseFeatureStore.
        
        Args:
            base_df: Optional initial DataFrame. If None, creates empty DataFrame.
        """
        if base_df is not None:
            self.base_df = base_df.copy()
        else:
            self.base_df = pd.DataFrame()
    
    def get_patient_from_db(self, hospital_name: str, unit_name: str, bed_no: str) -> Optional[dict]:
        """
        Gets the patient object from production secondary1 db using hospital_name, unit_name, and bed_no as filters

        Args:
            hospital_name: string, Hospital name of the patient
            unit_name: string, Unit name of the patient
            bed_no: string or int, Bed number of the patient

        Returns:
            patient json object or None if not found
        """
        dburi = os.environ["db_uri"]
        client = MongoClient(dburi)
        dbName = dburi.split('/')[-1].split('?')[0]

        db = client[dbName]
        patients = db["patients"]

        pat_obj = patients.find_one(
            {'hospitalName': hospital_name,
             'unitName': unit_name,
             'bedNo': bed_no,
            }, ['_id', 'covidDetails', 'orders', 'io', 'roxIndex', 'initialSymtoms', 'allergies', 'chiefComplaint', 'otherComplications', 'signsAtAdmission', 'symptomsAtAdmission', 'underlyingMedicalConditions', 'pastMedicalHistory', 'chronic', 'immune', 'diagnoses', 'markedToWriteNotes', 'communicatedOrders', 'pressors', 'pastPatientMonitorIds', 'apacheScore', 'ventFreeDays', 'name', 'lastName', 'onSet', 'patientImage', 'heightCm', 'weightKg', 'IBW', 'BMI', 'MRN', 'CPMRN', 'bedNo', 'camera', 'hospitalName', 'unitName', 'hospitalLogo', 'ICUAdmitDate', 'isCurrentlyAdmitted', 'encounters', 'isolation', 'createdBy', 'commandCenterID', 'hospitalID', 'unitID', 'transferHistory', 'documents', 'days', 'vitals', 'sbar', 'summary', 'code_blue', 'createdAt', 'updatedAt', '__v', 'lastOpened', 'notes', 'ICUDischargeDate', 'age', 'dob', 'sex', 'address', 'PCP', 'motherInfo', 'weightKg', 'weightUnit', 'gestationAge', 'birthWeight', 'birthWeightUnit', 'weightHistory', 'severity'])

        return pat_obj

    def get_all_currently_admitted_patients(self) -> List[dict]:
        """
        Get all patients from MongoDB where isCurrentlyAdmitted is True.
        
        Returns:
            List of patient json objects
        """
        logger.info("Fetching all currently admitted patients")
        
        dburi = os.environ["db_uri"]
        client = MongoClient(dburi)
        dbName = dburi.split('/')[-1].split('?')[0]
        
        db = client[dbName]
        patients = db["patients"]
        
        # Query for all currently admitted patients
        query = {'isCurrentlyAdmitted': True}
        
        # Use the same projection as get_patient_from_db
        projection = ['_id', 'covidDetails', 'orders', 'io', 'roxIndex', 'initialSymtoms', 'allergies', 
                    'chiefComplaint', 'otherComplications', 'signsAtAdmission', 'symptomsAtAdmission', 
                    'underlyingMedicalConditions', 'pastMedicalHistory', 'chronic', 'immune', 'diagnoses', 
                    'markedToWriteNotes', 'communicatedOrders', 'pressors', 'pastPatientMonitorIds', 
                    'apacheScore', 'ventFreeDays', 'name', 'lastName', 'onSet', 'patientImage', 
                    'heightCm', 'weightKg', 'IBW', 'BMI', 'MRN', 'CPMRN', 'bedNo', 'camera', 
                    'hospitalName', 'unitName', 'hospitalLogo', 'ICUAdmitDate', 'isCurrentlyAdmitted', 
                    'encounters', 'isolation', 'createdBy', 'commandCenterID', 'hospitalID', 'unitID', 
                    'transferHistory', 'documents', 'days', 'vitals', 'sbar', 'summary', 'code_blue', 
                    'createdAt', 'updatedAt', '__v', 'lastOpened', 'notes', 'ICUDischargeDate', 'age', 
                    'dob', 'sex', 'address', 'PCP', 'motherInfo', 'weightKg', 'weightUnit', 
                    'gestationAge', 'birthWeight', 'birthWeightUnit', 'weightHistory', 'severity']
        
        # Use find() instead of find_one() to get all matching patients
        pat_objs = list(patients.find(query, projection))
        
        logger.info(f"Found {len(pat_objs)} currently admitted patients")
        
        return pat_objs
    
    def getNotesKeys(self, key_name: str, df: pd.DataFrame, new_col_name: str) -> pd.DataFrame:
        """
        Extract values from notes.finalNotes based on displayName matching key_name.
        
        For each row, accesses notes.finalNotes column (list of dicts), checks if any dict has
        content.components.displayName matching key_name, and collects the corresponding
        content.components.value into a list.
        
        Args:
            key_name: The displayName to match against
            df: DataFrame to process
            new_col_name: Name of the new column to add
            
        Returns:
            pd.DataFrame: DataFrame with the new column added
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        def extract_values(row: pd.Series) -> List[Any]:
            """Extract values from notes.finalNotes for a single row."""
            values_list = []
            
            # Access notes.finalNotes column directly
            try:
                if 'notes.finalNotes' not in row.index:
                    return values_list
                
                final_notes = row['notes.finalNotes']
                
                # Check if final_notes is a list
                if not isinstance(final_notes, list):
                    logger.warning(f"final_notes is not a list in row {row.name}")
                    return values_list
                
                # Iterate through each dict in the list
                for item in final_notes:
                    if not isinstance(item, dict):
                        logger.warning(f"item is not a dict in row {row.name}")
                        continue
                    
                    # Navigate: item['content']['components']['displayName'] and ['value']
                    try:
                        content = item.get('content')
                        if not isinstance(content, dict):
                            if isinstance(content, list):
                                content = content[0]
                            else:
                                logger.warning(f"content is not a dict in row {row.name}")
                                continue
                        
                        components = content.get('components')
                        if components is None:
                            continue
                        
                        # Handle components as a dict
                        if isinstance(components, dict):
                            display_name = components.get('displayName')
                            if display_name == key_name:
                                value = components.get('value')
                                if value is not None:
                                    values_list.append(value)
                        # Handle components as a list of dicts
                        elif isinstance(components, list):
                            for comp in components:
                                if isinstance(comp, dict):
                                    display_name = comp.get('displayName')
                                    if display_name == key_name:
                                        value = comp.get('value')
                                        if value is not None:
                                            values_list.append(value)
                    except (KeyError, TypeError, AttributeError):
                        continue
            
            except (KeyError, TypeError, AttributeError):
                logger.warning(f"Error extracting values for {key_name} in row {row.name}")
                pass
            
            return values_list
        
        # Apply the extraction function to each row
        result_df[new_col_name] = result_df.apply(extract_values, axis=1)
        
        return result_df
    
    def create_base_df_from_patient(self, hospital_name: str, unit_name: str, bed_no: str) -> pd.DataFrame:
        """
        Create base DataFrame from a single patient record.
        
        Args:
            hospital_name: Hospital name of the patient
            unit_name: Unit name of the patient
            bed_no: Bed number of the patient
            
        Returns:
            pd.DataFrame: Base DataFrame with patient data, or empty DataFrame if patient not found
        """
        patient_obj = self.get_patient_from_db(hospital_name, unit_name, str(bed_no))
        
        if patient_obj is None:
            logger.warning(f"Patient not found for Hospital: {hospital_name}, Unit: {unit_name}, Bed: {bed_no}")
            return pd.DataFrame()
        
        # Convert MongoDB types (ObjectId, datetime) to serializable formats
        serializable_obj = convert_to_serializable(patient_obj)
        
        # Use json_normalize to flatten nested structures into a DataFrame
        df = json_normalize(serializable_obj)
        
        # Extract notes keys
        df = self.getNotesKeys('Diagnosis', df, 'notesDiagnoses')
        df = self.getNotesKeys('Summary', df, 'notesSummary')
        
        return df
    
    def add_patient(self, hospital_name: str, unit_name: str, bed_no: str) -> bool:
        """
        Add a patient's data to the base_df.
        
        Args:
            hospital_name: Hospital name of the patient
            unit_name: Unit name of the patient
            bed_no: Bed number of the patient
            
        Returns:
            bool: True if patient was added successfully, False otherwise
        """
        patient_df = self.create_base_df_from_patient(hospital_name, unit_name, bed_no)
        
        if patient_df.empty:
            return False
        
        # Append to base_df
        if self.base_df.empty:
            self.base_df = patient_df
        else:
            self.base_df = pd.concat([self.base_df, patient_df], ignore_index=True)
        
        logger.info(f"Added patient: Hospital: {hospital_name}, Unit: {unit_name}, Bed: {bed_no}")
        return True
    
    def load_from_csv(self, csv_file_path: str, merge_with_output: bool = True) -> int:
        """
        Load multiple patients from a CSV file and add them to base_df.
        
        CSV should have columns: hospital_name, unit_name, bed_no
        
        Args:
            csv_file_path: Path to CSV file
            merge_with_output: If True, after adding patients, merge CSV with base_df on 
                              hospital_name, unit_name, and bed_no to combine additional CSV columns.
            
        Returns:
            int: Number of patients successfully loaded
        """
        try:
            csv_df = pd.read_csv(csv_file_path)
            
            # Check if required columns exist
            required_cols = ['hospital_name', 'unit_name', 'bed_no']
            missing_cols = [col for col in required_cols if col not in csv_df.columns]
            if missing_cols:
                raise ValueError(f"CSV file is missing required columns: {missing_cols}")
            
            # Process each patient and add to base_df
            count = 0
            for _, row in csv_df.iterrows():
                hospital_name = row['hospital_name']
                unit_name = row['unit_name']
                bed_no = row['bed_no']
                
                if self.add_patient(hospital_name, unit_name, bed_no):
                    count += 1
            
            # If merge_with_output is True, merge CSV with base_df after adding patients
            if merge_with_output and not self.base_df.empty:
                # Check if base_df has the required columns for merging
                base_cols = ['hospitalName', 'unitName', 'bedNo']
                if all(col in self.base_df.columns for col in base_cols):
                    # Convert bed_no to string in CSV for consistent merging
                    csv_df_merge = csv_df.copy()
                    csv_df_merge['bed_no'] = csv_df_merge['bed_no'].astype(str)
                    
                    # Convert bedNo to string in base_df for consistent merging
                    base_df_merge = self.base_df.copy()
                    base_df_merge['bedNo'] = base_df_merge['bedNo'].astype(str)

                    # Merge CSV with base_df on the key columns
                    merged_df = base_df_merge.merge(
                        csv_df_merge,
                        left_on=['hospitalName', 'unitName', 'bedNo'],
                        right_on=['hospital_name', 'unit_name', 'bed_no'],
                        how='left',
                        suffixes=('', '_csv')
                    )
                    
                    # Drop the duplicate key columns from CSV merge
                    merged_df = merged_df.drop(columns=['hospital_name', 'unit_name', 'bed_no'], errors='ignore')
                    
                    # Update base_df with merged data
                    self.base_df = merged_df
                    logger.info(f"Merged CSV data with base_df for {len(csv_df)} patients")
                else:
                    logger.warning("base_df does not have required columns for merging, skipping merge step")
            
            logger.info(f"Loaded {count} out of {len(csv_df)} patients from {csv_file_path}")
            return count
            
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
            return 0
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}", exc_info=True)
            return 0
    
    def save_base_df(self, csv_path: str = 'BaseFeatureStore.csv') -> None:
        """
        Save base_df to CSV file.
        
        Args:
            csv_path: Path to save the CSV file
        """
        if self.base_df.empty:
            logger.warning("base_df is empty, nothing to save")
            return
        
        self.base_df.to_csv(csv_path, index=False)
        logger.info(f"Saved base features to: {csv_path}")
    
    def get_base_df(self) -> pd.DataFrame:
        """
        Get the base DataFrame.
        
        Returns:
            pd.DataFrame: Copy of the base_df
        """
        return self.base_df.copy()
