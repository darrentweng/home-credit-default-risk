"""
Data processing module for Home Credit Default Risk prediction.
Handles loading and cleaning of raw CSV files.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected CSV files in the data directory
EXPECTED_FILES = {
    'application_train': 'application_train.csv',
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv',
    'installments_payments': 'installments_payments.csv',
    'credit_card_balance': 'credit_card_balance.csv'
}


def load_all_data(data_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    Load all required CSV files from the data directory.
    
    Args:
        data_dir (str): Path to the directory containing CSV files
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with consistent naming convention
        
    Raises:
        FileNotFoundError: If required CSV files are missing
        pd.errors.EmptyDataError: If CSV files are empty
        pd.errors.ParserError: If CSV files are corrupted
    """
    data_dict = {}
    missing_files = []
    
    logger.info(f"Loading data from directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")
    
    # Load each expected file
    for key, filename in EXPECTED_FILES.items():
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            missing_files.append(filename)
            continue
            
        try:
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Warning: {filename} is empty")
                
            data_dict[key] = df
            logger.info(f"Successfully loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except pd.errors.EmptyDataError:
            # Handle completely empty files by creating an empty DataFrame
            logger.warning(f"Warning: {filename} is completely empty, creating empty DataFrame")
            data_dict[key] = pd.DataFrame()
            continue
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Error parsing {filename}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error loading {filename}: {str(e)}")
    
    # Report missing files
    if missing_files:
        missing_files_str = ", ".join(missing_files)
        error_msg = f"Missing required CSV files: {missing_files_str}. Please ensure all files are present in the '{data_dir}' directory."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Successfully loaded all {len(data_dict)} data files")
    return data_dict

def clean_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and preprocess all loaded DataFrames.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of raw DataFrames
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cleaned DataFrames
    """
    cleaned_data = {}
    
    logger.info("Starting data cleaning process...")
    
    for key, df in data_dict.items():
        logger.info(f"Cleaning {key} data...")
        cleaned_df = df.copy()
        
        if key == 'application_train':
            cleaned_df = _clean_application_data(cleaned_df)
        elif key == 'bureau':
            cleaned_df = _clean_bureau_data(cleaned_df)
        elif key == 'bureau_balance':
            cleaned_df = _clean_bureau_balance_data(cleaned_df)
        elif key == 'previous_application':
            cleaned_df = _clean_previous_application_data(cleaned_df)
        elif key == 'installments_payments':
            cleaned_df = _clean_installments_data(cleaned_df)
        elif key == 'credit_card_balance':
            cleaned_df = _clean_credit_card_data(cleaned_df)
        
        cleaned_data[key] = cleaned_df
        logger.info(f"Completed cleaning {key}: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    
    logger.info("Data cleaning process completed")
    return cleaned_data


def _clean_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean application train data with comprehensive data type handling."""
    df = df.copy()
    
    logger.info("Cleaning application data...")
    
    # Handle categorical columns - fill missing with 'Unknown'
    categorical_cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype('category')
    
    # Handle numerical columns - use median imputation and ensure proper data types
    numerical_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START'
    ]
    
    for col in numerical_cols:
        if col in df.columns:
            # Convert to numeric first, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Use median, but fallback to 0 if all values are NaN
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
            
            # Ensure integer columns are properly typed
            if col in ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 
                      'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START']:
                df[col] = df[col].astype('int64')
    
    # Handle days columns (negative values are normal - days before application)
    days_cols = [
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE'
    ]
    
    for col in days_cols:
        if col in df.columns:
            # Convert to numeric and handle known data errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle DAYS_EMPLOYED anomaly (365243 is a known data error)
            if col == 'DAYS_EMPLOYED':
                df[col] = df[col].replace(365243, np.nan)
            
            # Use median, but fallback to 0 if all values are NaN
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
            df[col] = df[col].astype('int64')
    
    # Handle flag columns (binary 0/1) - fill with 0 and ensure integer type
    flag_cols = [col for col in df.columns if col.startswith('FLAG_')]
    for col in flag_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0).astype('int8')
    
    # Handle region flag columns (binary 0/1)
    region_flag_cols = [col for col in df.columns if col.startswith('REG_') or col.startswith('LIVE_')]
    for col in region_flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int8')
    
    # Handle external source columns - fill with median
    ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    for col in ext_source_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Use median, but fallback to 0.5 if all values are NaN (normalized scores)
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.5
            df[col] = df[col].fillna(median_val)
    
    # Handle building information columns - fill with median for numerical
    building_cols = [col for col in df.columns if any(suffix in col for suffix in ['_AVG', '_MODE', '_MEDI'])]
    for col in building_cols:
        if col in df.columns:
            if any(suffix in col for suffix in ['_MODE']) and col not in ['TOTALAREA_MODE']:
                # Categorical building columns
                df[col] = df[col].fillna('Unknown').astype('category')
            else:
                # Numerical building columns
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Use median, but fallback to 0 if all values are NaN
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
    
    # Handle social circle columns - fill with 0
    social_cols = [col for col in df.columns if 'CNT_SOCIAL_CIRCLE' in col]
    for col in social_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Handle credit bureau inquiry columns - fill with 0
    credit_bureau_cols = [col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
    for col in credit_bureau_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0).astype('int64')
    
    # Handle document flag columns - fill with 0
    document_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
    for col in document_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0).astype('int8')
    
    # Ensure SK_ID_CURR is integer
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    
    # Ensure TARGET is integer (if present)
    if 'TARGET' in df.columns:
        df['TARGET'] = df['TARGET'].astype('int8')
    
    logger.info(f"Application data cleaning completed. Shape: {df.shape}")
    return df


def _clean_bureau_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean bureau data with proper data type handling."""
    df = df.copy()
    
    logger.info("Cleaning bureau data...")
    
    # Handle categorical columns
    categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype('category')
    
    # Handle numerical columns - for bureau data, 0 is often meaningful for missing values
    numerical_cols = [
        'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
        'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY'
    ]
    
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
            
            # Ensure integer columns are properly typed
            if col in ['CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG']:
                df[col] = df[col].astype('int64')
    
    # Handle days columns
    days_cols = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']
    for col in days_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Ensure ID columns are integers
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    if 'SK_ID_BUREAU' in df.columns:
        df['SK_ID_BUREAU'] = df['SK_ID_BUREAU'].astype('int64')
    
    logger.info(f"Bureau data cleaning completed. Shape: {df.shape}")
    return df


def _clean_bureau_balance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean bureau balance data with proper data type handling."""
    df = df.copy()
    
    logger.info("Cleaning bureau balance data...")
    
    # STATUS column - categorical with specific meanings
    if 'STATUS' in df.columns:
        df['STATUS'] = df['STATUS'].fillna('X').astype('category')  # X means unknown status
    
    # MONTHS_BALANCE - integer representing months relative to application
    if 'MONTHS_BALANCE' in df.columns:
        df['MONTHS_BALANCE'] = pd.to_numeric(df['MONTHS_BALANCE'], errors='coerce')
        df['MONTHS_BALANCE'] = df['MONTHS_BALANCE'].fillna(0).astype('int64')
    
    # Ensure ID columns are integers
    if 'SK_ID_BUREAU' in df.columns:
        df['SK_ID_BUREAU'] = df['SK_ID_BUREAU'].astype('int64')
    
    logger.info(f"Bureau balance data cleaning completed. Shape: {df.shape}")
    return df


def _clean_previous_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean previous application data with comprehensive data type handling."""
    df = df.copy()
    
    logger.info("Cleaning previous application data...")
    
    # Handle categorical columns
    categorical_cols = [
        'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
        'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype('category')
    
    # Handle numerical columns with appropriate imputation strategies
    amount_cols = [
        'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
        'AMT_GOODS_PRICE', 'SELLERPLACE_AREA'
    ]
    
    for col in amount_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Use median, but fallback to 0 if all values are NaN
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
    
    # Handle rate columns (percentages)
    rate_cols = ['RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
    for col in rate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Use median, but fallback to 0.15 (15%) if all values are NaN
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.15
            df[col] = df[col].fillna(median_val)
    
    # Handle integer columns
    integer_cols = ['HOUR_APPR_PROCESS_START', 'CNT_PAYMENT']
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Use median, but fallback to appropriate defaults if all values are NaN
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 12 if col == 'HOUR_APPR_PROCESS_START' else 12  # Default to noon or 12 payments
            df[col] = df[col].fillna(median_val).astype('int64')
    
    # Handle days columns
    days_cols = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 
                 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Ensure ID columns are integers
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    if 'SK_ID_PREV' in df.columns:
        df['SK_ID_PREV'] = df['SK_ID_PREV'].astype('int64')
    
    logger.info(f"Previous application data cleaning completed. Shape: {df.shape}")
    return df


def _clean_installments_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean installments payments data with proper data type handling."""
    df = df.copy()
    
    logger.info("Cleaning installments payments data...")
    
    # Handle integer columns
    integer_cols = ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Handle days columns
    days_cols = ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']
    for col in days_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Handle amount columns
    amount_cols = ['AMT_INSTALMENT', 'AMT_PAYMENT']
    for col in amount_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)  # 0 payment is meaningful
    
    # Ensure ID columns are integers
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    if 'SK_ID_PREV' in df.columns:
        df['SK_ID_PREV'] = df['SK_ID_PREV'].astype('int64')
    
    logger.info(f"Installments payments data cleaning completed. Shape: {df.shape}")
    return df


def _clean_credit_card_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean credit card balance data with comprehensive data type handling."""
    df = df.copy()
    
    logger.info("Cleaning credit card balance data...")
    
    # Handle categorical columns
    if 'NAME_CONTRACT_STATUS' in df.columns:
        df['NAME_CONTRACT_STATUS'] = df['NAME_CONTRACT_STATUS'].fillna('Unknown').astype('category')
    
    # Handle integer columns
    integer_cols = [
        'MONTHS_BALANCE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM',
        'SK_DPD', 'SK_DPD_DEF'
    ]
    
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int64')
    
    # Handle amount columns (float)
    amount_cols = [
        'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
        'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE'
    ]
    
    for col in amount_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)  # 0 is meaningful for credit card balances
    
    # Ensure ID columns are integers
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    if 'SK_ID_PREV' in df.columns:
        df['SK_ID_PREV'] = df['SK_ID_PREV'].astype('int64')
    
    logger.info(f"Credit card balance data cleaning completed. Shape: {df.shape}")
    return df