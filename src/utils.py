"""
Utility functions and helpers for Home Credit Default Risk prediction.
Common utility functions and constants.
"""

import pandas as pd
import numpy as np
import shap
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration Variables
RANDOM_STATE = 42

# Expected CSV files in the data directory
EXPECTED_DATA_FILES = {
    'application_train': 'application_train.csv',
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv',
    'installments_payments': 'installments_payments.csv',
    'credit_card_balance': 'credit_card_balance.csv'
}

# LightGBM model parameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'n_estimators': 1000,
    'is_unbalance': True
}

# Risk level thresholds for prediction categorization
RISK_THRESHOLDS = {
    'low': 0.3,      # Below 30% probability = Low risk
    'medium': 0.7,   # 30-70% probability = Medium risk
    'high': 1.0      # Above 70% probability = High risk
}

# Feature groups for better organization and interpretation
FEATURE_GROUPS = {
    'application': [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ],
    'bureau': [
        'bureau_credit_count', 'bureau_avg_credit_amount', 'bureau_max_overdue_amount',
        'bureau_credit_utilization_ratio', 'bureau_active_credit_count',
        'bureau_closed_credit_count', 'bureau_avg_days_credit', 'bureau_total_debt'
    ],
    'previous_applications': [
        'prev_app_count', 'prev_app_avg_credit', 'prev_app_approval_rate',
        'prev_app_avg_annuity', 'prev_app_avg_goods_price', 'prev_app_refused_count',
        'prev_app_approved_count', 'prev_app_avg_days_decision'
    ],
    'installments': [
        'installments_count', 'installments_payment_ratio', 'installments_late_payment_count',
        'installments_avg_payment_amount', 'installments_avg_instalment_amount',
        'installments_payment_consistency', 'installments_avg_days_late', 'installments_total_paid'
    ],
    'credit_card': [
        'cc_balance_count', 'cc_avg_balance_utilization', 'cc_payment_behavior_score',
        'cc_avg_balance', 'cc_avg_credit_limit', 'cc_avg_payment_amount',
        'cc_total_drawings', 'cc_avg_drawings_atm', 'cc_avg_drawings_pos',
        'cc_payment_consistency', 'cc_avg_dpd', 'cc_max_dpd', 'cc_active_months'
    ]
}


def get_feature_names() -> List[str]:
    """
    Returns standardized feature names used in the model.
    
    This function provides a centralized way to access all feature names
    used in the Home Credit Default Risk model, organized by feature groups.
    
    Returns:
        List[str]: List of all feature names used in the model
    
    Example:
        >>> features = get_feature_names()
        >>> print(f"Total features: {len(features)}")
        >>> print(f"First 5 features: {features[:5]}")
    """
    all_features = []
    
    # Combine all feature groups
    for group_name, features in FEATURE_GROUPS.items():
        all_features.extend(features)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for feature in all_features:
        if feature not in seen:
            seen.add(feature)
            unique_features.append(feature)
    
    logger.info(f"Retrieved {len(unique_features)} standardized feature names")
    return unique_features


def validate_data_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains the expected columns and data types.
    
    This function checks if the provided DataFrame has all the required columns
    and validates basic data integrity for the Home Credit Default Risk model.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        expected_columns (List[str]): List of column names that should be present
    
    Returns:
        bool: True if schema is valid, False otherwise
    
    Raises:
        ValueError: If DataFrame is empty or None
    
    Example:
        >>> df = pd.DataFrame({'SK_ID_CURR': [1, 2], 'TARGET': [0, 1]})
        >>> is_valid = validate_data_schema(df, ['SK_ID_CURR', 'TARGET'])
        >>> print(f"Schema valid: {is_valid}")
    """
    if df is None:
        logger.error("DataFrame is None")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    # Check for required columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Validate SK_ID_CURR if present (should be unique identifier)
    if 'SK_ID_CURR' in df.columns:
        if df['SK_ID_CURR'].isnull().any():
            logger.error("SK_ID_CURR contains null values")
            return False
        
        if df['SK_ID_CURR'].duplicated().any():
            logger.warning("SK_ID_CURR contains duplicate values")
            # Don't return False as duplicates might be valid in some contexts
    
    # Validate TARGET if present (should be binary 0/1)
    if 'TARGET' in df.columns:
        unique_targets = df['TARGET'].dropna().unique()
        if not all(target in [0, 1] for target in unique_targets):
            logger.error(f"TARGET column contains invalid values: {unique_targets}")
            return False
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        logger.warning(f"Columns with all null values: {empty_columns}")
    
    # Validate data types for key columns
    if 'SK_ID_CURR' in df.columns:
        if not pd.api.types.is_integer_dtype(df['SK_ID_CURR']):
            logger.warning("SK_ID_CURR is not integer type")
    
    if 'TARGET' in df.columns:
        if not pd.api.types.is_integer_dtype(df['TARGET']):
            logger.warning("TARGET is not integer type")
    
    # Check for reasonable data ranges
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col.startswith('AMT_') and (df[col] < 0).any():
            logger.warning(f"Amount column {col} contains negative values")
        
        if col.startswith('DAYS_') and (df[col] > 0).any():
            logger.warning(f"Days column {col} contains positive values (should be negative)")
    
    logger.info(f"Schema validation completed for DataFrame with shape {df.shape}")
    return True


def calculate_shap_values(model, X: pd.DataFrame, max_samples: Optional[int] = 1000) -> shap.Explanation:
    """
    Calculate SHAP values for model explanations.
    
    This function provides a wrapper around SHAP library to calculate
    feature importance and explanations for the LightGBM model predictions.
    
    Args:
        model: Trained LightGBM model
        X (pd.DataFrame): Feature data for SHAP calculation
        max_samples (Optional[int]): Maximum number of samples to use for SHAP calculation
                                   to manage computational complexity. Defaults to 1000.
    
    Returns:
        shap.Explanation: SHAP explanation object containing values and expected values
    
    Raises:
        ValueError: If model or data is invalid
        Exception: If SHAP calculation fails
    
    Example:
        >>> shap_values = calculate_shap_values(model, X_test, max_samples=500)
        >>> print(f"SHAP values shape: {shap_values.values.shape}")
        >>> print(f"Base value: {shap_values.base_values[0]}")
    """
    if model is None:
        raise ValueError("Model cannot be None")
    
    if X is None or X.empty:
        raise ValueError("Feature data cannot be None or empty")
    
    try:
        # Limit samples for computational efficiency
        if max_samples and len(X) > max_samples:
            logger.info(f"Sampling {max_samples} rows from {len(X)} for SHAP calculation")
            X_sample = X.sample(n=max_samples, random_state=RANDOM_STATE)
        else:
            X_sample = X.copy()
        
        # Handle categorical features by encoding them the same way as training
        X_processed = X_sample.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Create SHAP explainer for tree-based models
        logger.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        logger.info(f"Calculating SHAP values for {len(X_processed)} samples...")
        shap_values = explainer(X_processed)
        
        logger.info(f"SHAP calculation completed. Shape: {shap_values.values.shape}")
        return shap_values
        
    except Exception as e:
        logger.error(f"SHAP calculation failed: {str(e)}")
        raise Exception(f"SHAP calculation failed: {str(e)}")


def get_risk_category(probability: float) -> str:
    """
    Categorize default probability into risk levels.
    
    Args:
        probability (float): Default probability between 0 and 1
    
    Returns:
        str: Risk category ('Low', 'Medium', or 'High')
    
    Example:
        >>> risk = get_risk_category(0.25)
        >>> print(f"Risk level: {risk}")  # Output: Risk level: Low
    """
    if probability < RISK_THRESHOLDS['low']:
        return 'Low'
    elif probability < RISK_THRESHOLDS['medium']:
        return 'Medium'
    else:
        return 'High'


def get_feature_group(feature_name: str) -> str:
    """
    Get the feature group for a given feature name.
    
    Args:
        feature_name (str): Name of the feature
    
    Returns:
        str: Feature group name or 'unknown' if not found
    
    Example:
        >>> group = get_feature_group('bureau_credit_count')
        >>> print(f"Feature group: {group}")  # Output: Feature group: bureau
    """
    for group_name, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group_name
    return 'unknown'


def format_currency(amount: float) -> str:
    """
    Format monetary amounts for display.
    
    Args:
        amount (float): Monetary amount
    
    Returns:
        str: Formatted currency string
    
    Example:
        >>> formatted = format_currency(150000.50)
        >>> print(formatted)  # Output: $150,000.50
    """
    return f"${amount:,.2f}"


def format_percentage(ratio: float) -> str:
    """
    Format ratios as percentages for display.
    
    Args:
        ratio (float): Ratio between 0 and 1
    
    Returns:
        str: Formatted percentage string
    
    Example:
        >>> formatted = format_percentage(0.1234)
        >>> print(formatted)  # Output: 12.34%
    """
    return f"{ratio * 100:.2f}%"


def get_model_config() -> Dict[str, Any]:
    """
    Get the standard LightGBM model configuration.
    
    Returns:
        Dict[str, Any]: Dictionary containing model parameters
    
    Example:
        >>> config = get_model_config()
        >>> print(f"Learning rate: {config['learning_rate']}")
    """
    return LIGHTGBM_PARAMS.copy()


def log_data_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log basic information about a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name for logging purposes
    
    Example:
        >>> log_data_info(df, "Training Data")
    """
    logger.info(f"{name} - Shape: {df.shape}")
    logger.info(f"{name} - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if not df.empty:
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"{name} - Columns with null values: {null_counts[null_counts > 0].to_dict()}")
        else:
            logger.info(f"{name} - No null values found")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"{name} - Numeric columns: {len(numeric_cols)}")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        logger.info(f"{name} - Categorical columns: {len(categorical_cols)}")