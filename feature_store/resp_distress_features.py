import pandas as pd
import re
import logging
from typing import Optional, Union, List, Dict, Any
import sys
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.insert(0, str(project_root))

from feature_store import BaseFeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def string_search(target: Union[str, List[str], None], key: str) -> bool:
    """
    Searches for a key string in a target string or list of strings (case insensitive).
    
    Args:
        target: String or list of strings to search in. Can be None.
        key: String to search for (exact match or substring, case insensitive)
        
    Returns:
        bool: True if key is found (exact match or substring) in target, False otherwise
    """
    # Handle None or empty key
    if key is None or not isinstance(key, str) or len(key) == 0:
        return False
    
    # Handle None target
    if target is None:
        return False
    
    # Convert key to lowercase for case-insensitive comparison
    key_lower = key.lower()
    
    # If target is a string, check if key is same or substring (case insensitive)
    if isinstance(target, str):
        return key_lower in target.lower()
    
    # If target is a list, check each item
    if isinstance(target, list):
        for item in target:
            # Skip None items
            if item is None:
                continue
            # Convert to string if not already (handles non-string items)
            item_str = str(item) if not isinstance(item, str) else item
            if key_lower in item_str.lower():
                return True
        return False
    
    # For other types, try to convert to string and search
    try:
        target_str = str(target)
        return key_lower in target_str.lower()
    except:
        return False


def check_relevant_disease(row: pd.Series, lookFor: List[str], Cols: List[str]) -> bool:
    """
    Check if any disease in lookForDiseases matches in any column in diseaseCols for this row.
    
    Supports square bracket notation for list of dicts access:
    - "column_name" -> accesses row[column_name]
    - "column_name[key]" -> expects row[column_name] to be a list of dicts,
                            extracts dict[key] from each dict in the list,
                            and searches in all extracted values
    
    Args:
        row: pandas Series representing a row in the DataFrame
        lookFor: List of disease/order strings to search for
        Cols: List of column names to search in. Can use square bracket notation (e.g., "col[key]")
        
    Returns:
        bool: True if any match is found, False otherwise
    """
    # If no diseases to look for or no columns to search, return False
    if not lookFor or not Cols:
        return False
    
    # Check each disease against each column
    for disease in lookFor:
        for col_spec in Cols:
            # Parse column specification (handle square bracket notation)
            # Pattern: "column_name[key]" -> col_name="column_name", key_name="key"
            bracket_match = re.search(r'^(.+)\[(.+)\]$', col_spec)
            if bracket_match:
                # Extract column name and key from square brackets
                col_name = bracket_match.group(1)
                key_name = bracket_match.group(2)
            else:
                # Regular column name (no brackets)
                col_name = col_spec
                key_name = None
            
            # Skip if column doesn't exist in the DataFrame
            if col_name not in row.index:
                continue
            
            # Get the value to search in
            try:
                if key_name is not None:
                    # row[col_name] is expected to be a list of dicts
                    # Extract key_name from each dict in the list
                    col_value = row[col_name]
                    
                    # Check if col_value is a list
                    if not isinstance(col_value, list):
                        continue
                    
                    # Extract values of key_name from each dict in the list
                    extracted_values = []
                    for item in col_value:
                        if not isinstance(item, dict):
                            continue
                        # Get the value of key_name from this dict
                        if key_name in item:
                            value = item[key_name]
                            # If value is a list, extend extracted_values with all items
                            # Otherwise, append the value
                            if isinstance(value, list):
                                extracted_values.extend(value)
                            else:
                                extracted_values.append(value)
                    
                    # If no values extracted, skip
                    if not extracted_values:
                        continue
                    
                    # Pass the list of extracted values to string_search
                    search_target = extracted_values
                else:
                    # Regular column access: row[col_name]
                    search_target = row[col_name]
                
                # Use string_search to check if disease matches in this column
                if string_search(search_target, disease):
                    return True
            except (KeyError, TypeError, AttributeError):
                # Skip if there's an error accessing the nested key
                continue
    
    return False


def check_num_labs(row: pd.Series, thresholds: Dict[str, List[Optional[float]]], label_type: str) -> bool:
    """
    Check numeric lab values against thresholds.
    
    Args:
        row: pandas Series representing a row in the DataFrame
        thresholds: Dictionary with keys as attribute names and values as [min, max] lists.
                    Either min or max can be None (but not both).
        label_type: The label type to filter documents by (e.g., 'ABG', 'CBC', etc.). .
    
    Returns:
        bool: True if any lab value is outside the threshold, False otherwise
    """
    # Check if documents column exists
    if 'documents' not in row.index:
        return False
    
    documents = row['documents']
    
    # Handle None or empty list
    if documents is None or not isinstance(documents, list) or len(documents) == 0:
        return False
    
    # Filter items where label matches the specified lab type
    lab_items = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        if item.get('label') == label_type:
            lab_items.append(item)
    
    # If no lab items found, return False
    if len(lab_items) == 0:
        return False
    
    # If multiple items, get the latest one based on reportedAt
    if len(lab_items) > 1:
        # Sort by reportedAt (latest first)
        try:
            lab_items.sort(
                key=lambda x: pd.to_datetime(x.get('reportedAt', '1900-01-01')),
                reverse=True
            )
        except (ValueError, TypeError):
            # If reportedAt parsing fails, use the first item
            pass
    
    # Get the latest lab item
    selected_item = lab_items[0]
    
    # Get the attributes dict
    attributes = selected_item.get('attributes')
    if attributes is None or not isinstance(attributes, dict):
        return False
    
    # Check each attribute value against thresholds
    for attr_key, attr_value in attributes.items():
        if not isinstance(attr_value, dict):
            continue
        
        # Get the 'value' key from the attribute dict
        value = attr_value.get('value')
        
        # Skip if value is None or not a number
        if value is None:
            continue
        
        try:
            value_float = float(value)
        except (ValueError, TypeError):
            continue
        
        # Check if this attribute has a threshold defined
        if attr_key in thresholds:
            min_val, max_val = thresholds[attr_key]
            
            # Handle None values for min or max
            # If min is None, only check upper bound
            # If max is None, only check lower bound
            # If both are None, skip this threshold check
            if min_val is None and max_val is None:
                continue
            elif min_val is None:
                # Only check upper bound
                if value_float > max_val:
                    return True
            elif max_val is None:
                # Only check lower bound
                if value_float < min_val:
                    return True
            else:
                # Both min and max are defined
                if value_float < min_val or value_float > max_val:
                    return True
    
    # All values are within thresholds
    return False

def check_ketosis(row: pd.Series) -> bool:
    """
    Check ketosis values against thresholds.
    
    Args:
        row: pandas Series representing a row in the DataFrame
    
    Returns:
        bool: True if ketosis is present, False otherwise
    """
    # Check if documents column exists
    if 'documents' not in row.index:
        return False
    
    documents = row['documents']
    
    # Handle None or empty list
    if documents is None or not isinstance(documents, list) or len(documents) == 0:
        return False
    
    # Filter items where label matches the specified lab type
    lab_items = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        if item.get('label').lower() == 'urine routine':
            lab_items.append(item)
    
    # If no lab items found, return False
    if len(lab_items) == 0:
        return False
    
    # If multiple items, get the latest one based on reportedAt
    if len(lab_items) > 1:
        # Sort by reportedAt (latest first)
        try:
            lab_items.sort(
                key=lambda x: pd.to_datetime(x.get('reportedAt', '1900-01-01')),
                reverse=True
            )
        except (ValueError, TypeError):
            # If reportedAt parsing fails, use the first item
            pass
    
    # Get the latest lab item
    selected_item = lab_items[0]
    
    # Get the attributes dict
    attributes = selected_item.get('attributes')
    if attributes is None or not isinstance(attributes, dict):
        return False
    
    ketosis_value = attributes.get('urine ketones')
    if ketosis_value is None or not ketosis_value:
        return False
    else:
        return True


class RespiratoryDistressFeatures:
    """
    Class to add respiratory distress features to a base DataFrame.
    
    Can be used standalone or with a BaseFeatureStore instance.
    """
    
    def __init__(self, base_df: Optional[pd.DataFrame] = None):
        """
        Initialize RespiratoryDistressFeatures.
        
        Args:
            base_df: Optional base DataFrame. Can be set later using set_base_df().
        """
        self.base_df = base_df if base_df is not None else pd.DataFrame()
        self.features_df = pd.DataFrame()
    
    def set_base_df(self, base_df: pd.DataFrame) -> None:
        """
        Set the base DataFrame.
        
        Args:
            base_df: Base DataFrame to add features to
        """
        self.base_df = base_df.copy() if base_df is not None else pd.DataFrame()
    
    def create_features(self, base_df: Optional[pd.DataFrame] = None, csv_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Creates respiratory distress features from patient data.
        
        Args:
            base_df: Optional DataFrame to process. If None, uses self.base_df.
            csv_file_path: Optional path to CSV file. If provided, all columns from CSV will be added to colsToKeep.
        
        Returns:
            pd.DataFrame: DataFrame with respiratory distress features added
        """
        # Use provided base_df or instance base_df
        if base_df is not None:
            df = base_df.copy()
        elif not self.base_df.empty:
            df = self.base_df.copy()
        else:
            return pd.DataFrame()
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # DiseaseFeature
        lookForDiagnoses = ['post surgery','Ashtma','bronchiectasis','pneumonia','lung abscess','interstitial lung disease','pulmonary edema','Diabetic ketosis','tachypneoa','orthopnea']
        diagnoseCols = ['signsAtAdmission','diagnoses','notesDiagnoses']
        
        # Add isRelevantDiagnose column
        df['isRelevantDiagnose'] = df.apply(
            lambda row: check_relevant_disease(row, lookForDiagnoses, diagnoseCols), 
            axis=1
        )

        lookForSymptoms = ['Breathlessness','shortness of breath','tiredness','chest pain','sudden onset sweating','abdominal pain','decreased urination','increased urination']
        symptomsCols = ['initialSymtoms','chiefComplaint','otherComplications','symptomsAtAdmission']
        
        # Add isRelevantSymptoms column
        df['isRelevantSymptoms'] = df.apply(
            lambda row: check_relevant_disease(row, lookForSymptoms, symptomsCols), 
            axis=1
        )

        lookForHistory = ['uncontrolled diabetes','heart disease','abdomen pain','sweating','trauma to chest','post surgery','major surgery','chest pain','breathlessness']
        historyCols = ['chronic','immune','notesSummary']
        
        # Add isRelevantHistory column
        df['isRelevantHistory'] = df.apply(
            lambda row: check_relevant_disease(row, lookForHistory, historyCols), 
            axis=1
        )    
        
        # OrderFeature
        lookForOrders = ['nebulizer','NIV','o2 face mask','nasal cannula','salbutamol','formetarol','inhalational hydrocortisone','aspirin','sodium bicarb infusion','soda bicarb','bicarb infusion','lasix','furosemide','intercostal drainage tube','insulin infusion']
        orderCols = ['orders.active.medications[name]','orders.active.labs[investigation]','orders.active.diets[name]','orders.active.procedures[name]','orders.active.bloods[title]','orders.active.vents[name]', 'orders.pending.medications[name]','orders.pending.labs[investigation]','orders.pending.diets[name]','orders.pending.procedures[name]','orders.pending.bloods[title]','orders.pending.vents[name]', 'orders.completed.medications[name]','orders.completed.labs[investigation]','orders.completed.diets[name]','orders.completed.procedures[name]','orders.completed.bloods[title]','orders.completed.vents[name]']
        
        # Add isRelevantOrder column
        df['isRelevantOrder'] = df.apply(
            lambda row: check_relevant_disease(row, lookForOrders, orderCols), 
            axis=1
        )

        #LabFeatures

        #ABGFeature
        abg_thresholds = {
            'pH': [7.35, None],    
            'HCO3': [20, None]
        }
        
        # Add isAbgAbnormal column
        df['isAbgAbnormal'] = df.apply(
            lambda row: check_num_labs(row, abg_thresholds, label_type='ABG'),
            axis=1
        )

        rbs_thresholds = {
            'rbs': [None,300]
        }
        # Add isRbsAbnormal column
        df['isRbsAbnormal'] = df.apply(
            lambda row: check_num_labs(row, rbs_thresholds, label_type='RBS'),
            axis=1
        )

        # Add isDiabeticKetosis column
        # True if both isRbsAbnormal is True and check_ketosis returns True
        df['isDiabeticKetosis'] = df.apply(
            lambda row: row['isRbsAbnormal'] == True and check_ketosis(row) == True,
            axis=1
        )

        colsToKeep = ['_id','name','lastName', 'initialSymtoms', 'allergies', 'chiefComplaint', 'otherComplications', 'onSet', 'signsAtAdmission', 'symptomsAtAdmission','chronic', 'immune','vitals', 'age.year', 'age.month', 'age.day', 'age.hour', 'age.minute', 'orders.active.medications', 'orders.active.labs', 'orders.active.diets', 'orders.active.bloods', 'orders.active.procedures', 'orders.active.vents', 'orders.pending.medications', 'orders.pending.labs', 'orders.pending.diets', 'orders.pending.communications', 'orders.pending.bloods', 'orders.pending.procedures', 'orders.pending.vents', 'orders.completed.medications', 'orders.completed.labs', 'orders.completed.diets', 'orders.completed.communications', 'orders.completed.bloods', 'orders.completed.procedures', 'orders.completed.vents', 'notes.finalNotes', 'isRelevantDiagnose','isRelevantSymptoms','isRelevantHistory','isRelevantOrder','isAbgAbnormal','isRbsAbnormal','isDiabeticKetosis']
        
        # Add all columns from the original CSV file if provided
        if csv_file_path:
            try:
                csv_df = pd.read_csv(csv_file_path, nrows=0)  # Read only headers
                csv_columns = csv_df.columns.tolist()
                for col in csv_columns:
                    if col not in colsToKeep:
                        colsToKeep.append(col)
                logger.info(f"Added {len(csv_columns)} columns from CSV file to colsToKeep")
            except Exception as e:
                logger.warning(f"Could not read CSV file to get columns: {e}")
        
        # Add missing columns as blank columns
        for col in colsToKeep:
            if col not in df.columns:
                df[col] = None
        
        # Select columns in the order specified by colsToKeep
        df = df[colsToKeep]
        
        # Store features_df
        self.features_df = df
        
        return df
    
    def get_features_df(self) -> pd.DataFrame:
        """
        Get the features DataFrame.
        
        Returns:
            pd.DataFrame: Copy of the features_df
        """
        return self.features_df.copy()
    
    def save_features_df(self, csv_path: str = 'RespiratoryDistressFeatures.csv') -> None:
        """
        Save features_df to CSV file.
        
        Args:
            csv_path: Path to save the CSV file
        """
        if self.features_df.empty:
            logger.warning("features_df is empty, nothing to save")
            return
        
        self.features_df.to_csv(csv_path, index=False)
        logger.info(f"Saved respiratory distress features to: {csv_path}")


# Standalone function for backward compatibility
def create_respiratory_distress_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates respiratory distress features from patient data.
    
    This is a wrapper function for backward compatibility.
    Uses the RespiratoryDistressFeatures class internally.
    
    Args:
        df (pd.DataFrame): DataFrame containing patient data with flattened structure
        
    Returns:
        pd.DataFrame: DataFrame with respiratory distress features added
    """
    feature_creator = RespiratoryDistressFeatures(df)
    return feature_creator.create_features()


def main():
    """
    Main function to process multiple patient records and generate feature stores.
    
    Reads patient identifiers from CSV, creates base feature store, adds respiratory distress features,
    and saves both base and features DataFrames to CSV files.
    
    Returns:
        tuple: (base_store, resp_features) - BaseFeatureStore and RespiratoryDistressFeatures instances
    """
    # CSV file path - should have columns: hospital_name, unit_name, bed_no
    csv_file_path = 'patient_list.csv'

    logger.info("Respiratory Distress Features Generation")
    
    # Create base feature store instance
    base_store = BaseFeatureStore()
    
    # Load patients from CSV
    logger.info(f"Loading patients from {csv_file_path}...")
    count = base_store.load_from_csv(csv_file_path)
    
    if count == 0:
        logger.warning("No patients loaded. Exiting.")
        return None, None
    
    # Get base DataFrame
    base_df = base_store.get_base_df()
    logger.info(f"Base DataFrame shape: {base_df.shape}")
    
    # Create respiratory distress features
    logger.info("Creating respiratory distress features...")
    resp_features = RespiratoryDistressFeatures(base_df)
    features_df = resp_features.create_features(csv_file_path=csv_file_path)
    logger.info(f"Features DataFrame shape: {features_df.shape}")
    
    # Save outputs
    logger.info("Saving outputs...")
    base_store.save_base_df('BaseFeatureStore.csv')
    resp_features.save_features_df('RespiratoryDistressFeatures.csv')

    logger.info("Processing Complete!")
    logger.info(f"Summary:")
    logger.info(f"  - Patients processed: {count}")
    logger.info(f"  - Base DataFrame: {base_df.shape}")
    logger.info(f"  - Features DataFrame: {features_df.shape}")
    logger.info(f"  - Output files:")
    logger.info(f"    * BaseFeatureStore.csv")
    logger.info(f"    * RespiratoryDistressFeatures.csv")
    
    return base_store, resp_features


if __name__ == "__main__":
    main()
