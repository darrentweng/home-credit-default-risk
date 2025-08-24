"""
Feature engineering module for Home Credit Default Risk prediction.
Creates predictive features from multiple data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def bureau_features(bureau_df: pd.DataFrame, bureau_balance_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create aggregated features from credit bureau data.
    Fixed to work with actual Kaggle Home Credit data structure.
    
    Args:
        bureau_df: DataFrame containing bureau credit information
        bureau_balance_df: Optional DataFrame containing bureau balance history
        
    Returns:
        DataFrame with aggregated bureau features by SK_ID_CURR
    """
    if bureau_df.empty:
        # Return empty DataFrame with expected columns if no bureau data
        return pd.DataFrame(columns=[
            'SK_ID_CURR', 'bureau_credit_count', 'bureau_avg_credit_amount',
            'bureau_max_overdue_amount', 'bureau_credit_utilization_ratio',
            'bureau_active_credit_count', 'bureau_closed_credit_count',
            'bureau_avg_days_credit', 'bureau_total_debt'
        ])
    
    print(f"Processing bureau data with shape: {bureau_df.shape}")
    print(f"Available columns: {list(bureau_df.columns)}")
    
    # Define aggregations based on available columns
    # Use the correct column name from Kaggle data
    agg_dict = {
        'SK_ID_BUREAU': 'count',  # Total number of bureau credits (correct column name)
    }
    
    # Add aggregations for columns that exist
    if 'AMT_CREDIT_SUM' in bureau_df.columns:
        agg_dict['AMT_CREDIT_SUM'] = ['mean', 'sum', 'max']
    if 'AMT_CREDIT_MAX_OVERDUE' in bureau_df.columns:
        agg_dict['AMT_CREDIT_MAX_OVERDUE'] = ['max', 'sum']
    if 'AMT_CREDIT_SUM_DEBT' in bureau_df.columns:
        agg_dict['AMT_CREDIT_SUM_DEBT'] = ['sum', 'mean']
    if 'AMT_CREDIT_SUM_LIMIT' in bureau_df.columns:
        agg_dict['AMT_CREDIT_SUM_LIMIT'] = ['sum', 'mean']
    if 'DAYS_CREDIT' in bureau_df.columns:
        agg_dict['DAYS_CREDIT'] = 'mean'
    if 'CREDIT_DAY_OVERDUE' in bureau_df.columns:
        agg_dict['CREDIT_DAY_OVERDUE'] = ['max', 'mean']
    if 'CNT_CREDIT_PROLONG' in bureau_df.columns:
        agg_dict['CNT_CREDIT_PROLONG'] = ['sum', 'mean']
    
    print(f"Aggregating with columns: {list(agg_dict.keys())}")
    
    try:
        bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    except Exception as e:
        print(f"Error during aggregation: {e}")
        print(f"Available columns in bureau_df: {list(bureau_df.columns)}")
        print(f"Trying to aggregate: {list(agg_dict.keys())}")
        raise
    
    # Flatten column names
    bureau_agg.columns = ['SK_ID_CURR'] + [
        f'bureau_{col[1]}_{col[0]}' if col[1] != '' else f'bureau_{col[0]}'
        for col in bureau_agg.columns[1:]
    ]
    
    # Rename key columns to match expected names
    column_mapping = {
        'bureau_count_SK_ID_BUREAU': 'bureau_credit_count',  # Fixed column name
        'bureau_mean_AMT_CREDIT_SUM': 'bureau_avg_credit_amount',
        'bureau_max_AMT_CREDIT_MAX_OVERDUE': 'bureau_max_overdue_amount',
        'bureau_sum_AMT_CREDIT_SUM_DEBT': 'bureau_total_debt',
        'bureau_mean_DAYS_CREDIT': 'bureau_avg_days_credit'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in bureau_agg.columns:
            bureau_agg = bureau_agg.rename(columns={old_name: new_name})
    
    # Calculate credit utilization ratio (debt / limit)
    if 'bureau_sum_AMT_CREDIT_SUM_LIMIT' in bureau_agg.columns and 'bureau_total_debt' in bureau_agg.columns:
        bureau_agg['bureau_credit_utilization_ratio'] = (
            bureau_agg['bureau_total_debt'] / 
            bureau_agg['bureau_sum_AMT_CREDIT_SUM_LIMIT'].replace(0, np.nan)
        ).fillna(0)
    else:
        bureau_agg['bureau_credit_utilization_ratio'] = 0
    
    # Count active vs closed credits
    if 'CREDIT_ACTIVE' in bureau_df.columns:
        credit_status = bureau_df.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack(fill_value=0)
        if 'Active' in credit_status.columns:
            bureau_agg = bureau_agg.merge(
                credit_status[['Active']].rename(columns={'Active': 'bureau_active_credit_count'}),
                on='SK_ID_CURR', how='left'
            )
        else:
            bureau_agg['bureau_active_credit_count'] = 0
            
        if 'Closed' in credit_status.columns:
            bureau_agg = bureau_agg.merge(
                credit_status[['Closed']].rename(columns={'Closed': 'bureau_closed_credit_count'}),
                on='SK_ID_CURR', how='left'
            )
        else:
            bureau_agg['bureau_closed_credit_count'] = 0
    else:
        bureau_agg['bureau_active_credit_count'] = 0
        bureau_agg['bureau_closed_credit_count'] = 0
    
    # Fill missing values
    numeric_columns = bureau_agg.select_dtypes(include=[np.number]).columns
    bureau_agg[numeric_columns] = bureau_agg[numeric_columns].fillna(0)
    
    return bureau_agg


def previous_application_features(prev_app_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from previous Home Credit applications.
    
    Args:
        prev_app_df: DataFrame containing previous application data
        
    Returns:
        DataFrame with aggregated previous application features by SK_ID_CURR
    """
    if prev_app_df.empty:
        # Return empty DataFrame with expected columns if no previous application data
        return pd.DataFrame(columns=[
            'SK_ID_CURR', 'prev_app_count', 'prev_app_avg_credit',
            'prev_app_approval_rate', 'prev_app_avg_annuity',
            'prev_app_avg_goods_price', 'prev_app_refused_count',
            'prev_app_approved_count', 'prev_app_avg_days_decision'
        ])
    
    print(f"Processing previous application data with shape: {prev_app_df.shape}")
    print(f"Available columns: {list(prev_app_df.columns)}")
    
    # Define aggregations based on available columns
    agg_dict = {
        'SK_ID_PREV': 'count',  # Total number of previous applications
    }
    
    # Add aggregations for columns that exist
    if 'AMT_CREDIT' in prev_app_df.columns:
        agg_dict['AMT_CREDIT'] = ['mean', 'sum', 'max']
    if 'AMT_ANNUITY' in prev_app_df.columns:
        agg_dict['AMT_ANNUITY'] = ['mean', 'sum']
    if 'AMT_APPLICATION' in prev_app_df.columns:
        agg_dict['AMT_APPLICATION'] = ['mean', 'sum']
    if 'AMT_GOODS_PRICE' in prev_app_df.columns:
        agg_dict['AMT_GOODS_PRICE'] = ['mean', 'sum']
    if 'DAYS_DECISION' in prev_app_df.columns:
        agg_dict['DAYS_DECISION'] = 'mean'
    if 'CNT_PAYMENT' in prev_app_df.columns:
        agg_dict['CNT_PAYMENT'] = ['mean', 'sum']
    
    print(f"Aggregating previous applications with columns: {list(agg_dict.keys())}")
    
    try:
        prev_app_agg = prev_app_df.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    except Exception as e:
        print(f"Error during previous application aggregation: {e}")
        print(f"Available columns: {list(prev_app_df.columns)}")
        raise
    
    # Flatten column names
    prev_app_agg.columns = ['SK_ID_CURR'] + [
        f'prev_app_{col[1]}_{col[0]}' if col[1] != '' else f'prev_app_{col[0]}'
        for col in prev_app_agg.columns[1:]
    ]
    
    # Rename key columns to match expected names
    column_mapping = {
        'prev_app_count_SK_ID_PREV': 'prev_app_count',
        'prev_app_mean_AMT_CREDIT': 'prev_app_avg_credit',
        'prev_app_mean_AMT_ANNUITY': 'prev_app_avg_annuity',
        'prev_app_mean_AMT_GOODS_PRICE': 'prev_app_avg_goods_price',
        'prev_app_mean_DAYS_DECISION': 'prev_app_avg_days_decision'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in prev_app_agg.columns:
            prev_app_agg = prev_app_agg.rename(columns={old_name: new_name})
    
    # Calculate approval rate if contract status is available
    if 'NAME_CONTRACT_STATUS' in prev_app_df.columns:
        # Count approved vs total applications
        status_counts = prev_app_df.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
        
        # Calculate approval rate
        if 'Approved' in status_counts.columns:
            prev_app_agg = prev_app_agg.merge(
                status_counts[['Approved']].rename(columns={'Approved': 'prev_app_approved_count'}),
                on='SK_ID_CURR', how='left'
            )
            prev_app_agg['prev_app_approval_rate'] = (
                prev_app_agg['prev_app_approved_count'] / prev_app_agg['prev_app_count']
            ).fillna(0)
        else:
            prev_app_agg['prev_app_approved_count'] = 0
            prev_app_agg['prev_app_approval_rate'] = 0
        
        # Count refused applications
        if 'Refused' in status_counts.columns:
            prev_app_agg = prev_app_agg.merge(
                status_counts[['Refused']].rename(columns={'Refused': 'prev_app_refused_count'}),
                on='SK_ID_CURR', how='left'
            )
        else:
            prev_app_agg['prev_app_refused_count'] = 0
    else:
        prev_app_agg['prev_app_approval_rate'] = 0
        prev_app_agg['prev_app_approved_count'] = 0
        prev_app_agg['prev_app_refused_count'] = 0
    
    # Fill missing values
    numeric_columns = prev_app_agg.select_dtypes(include=[np.number]).columns
    prev_app_agg[numeric_columns] = prev_app_agg[numeric_columns].fillna(0)
    
    return prev_app_agg


def installments_features(installments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from installment payment history.
    
    Args:
        installments_df: DataFrame containing installment payment data
        
    Returns:
        DataFrame with aggregated installment features by SK_ID_CURR
    """
    if installments_df.empty:
        # Return empty DataFrame with expected columns if no installment data
        return pd.DataFrame(columns=[
            'SK_ID_CURR', 'installments_count', 'installments_payment_ratio',
            'installments_late_payment_count', 'installments_avg_payment_amount',
            'installments_avg_instalment_amount', 'installments_payment_consistency',
            'installments_avg_days_late', 'installments_total_paid'
        ])
    
    # Calculate payment ratios and late payments
    if 'AMT_PAYMENT' in installments_df.columns and 'AMT_INSTALMENT' in installments_df.columns:
        installments_df = installments_df.copy()
        # Calculate payment ratio (actual payment / expected payment)
        installments_df['payment_ratio'] = (
            installments_df['AMT_PAYMENT'] / 
            installments_df['AMT_INSTALMENT'].replace(0, np.nan)
        ).fillna(0)
        
        # Calculate days late (positive means late payment)
        # Since days are relative to application (negative values), 
        # a later payment has a less negative value (closer to 0)
        if 'DAYS_ENTRY_PAYMENT' in installments_df.columns and 'DAYS_INSTALMENT' in installments_df.columns:
            installments_df['days_late'] = (
                installments_df['DAYS_ENTRY_PAYMENT'] - installments_df['DAYS_INSTALMENT']
            )
            # For negative relative days, late payment means DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
            # (less negative = later in time)
            # However, we need to handle the case where both are negative properly
            # If DAYS_ENTRY_PAYMENT is greater (less negative) than DAYS_INSTALMENT, it's late
            installments_df['is_late'] = (installments_df['days_late'] > 0).astype(int)
        else:
            installments_df['days_late'] = 0
            installments_df['is_late'] = 0
    else:
        installments_df = installments_df.copy()
        installments_df['payment_ratio'] = 0
        installments_df['days_late'] = 0
        installments_df['is_late'] = 0
    
    print(f"Processing installments data with shape: {installments_df.shape}")
    print(f"Available columns: {list(installments_df.columns)}")
    
    # Define aggregations based on available columns
    agg_dict = {
        'NUM_INSTALMENT_NUMBER': 'count',  # Total number of installments
    }
    
    # Add aggregations for columns that exist
    if 'AMT_PAYMENT' in installments_df.columns:
        agg_dict['AMT_PAYMENT'] = ['mean', 'sum', 'std']
    if 'AMT_INSTALMENT' in installments_df.columns:
        agg_dict['AMT_INSTALMENT'] = ['mean', 'sum']
    if 'payment_ratio' in installments_df.columns:
        agg_dict['payment_ratio'] = ['mean', 'std']
    if 'days_late' in installments_df.columns:
        agg_dict['days_late'] = ['mean', 'max']
    if 'is_late' in installments_df.columns:
        agg_dict['is_late'] = 'sum'
    
    print(f"Aggregating installments with columns: {list(agg_dict.keys())}")
    
    try:
        installments_agg = installments_df.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    except Exception as e:
        print(f"Error during installments aggregation: {e}")
        print(f"Available columns: {list(installments_df.columns)}")
        raise
    
    # Flatten column names
    installments_agg.columns = ['SK_ID_CURR'] + [
        f'installments_{col[1]}_{col[0]}' if col[1] != '' else f'installments_{col[0]}'
        for col in installments_agg.columns[1:]
    ]
    
    # Rename key columns to match expected names
    column_mapping = {
        'installments_count_NUM_INSTALMENT_NUMBER': 'installments_count',
        'installments_mean_payment_ratio': 'installments_payment_ratio',
        'installments_sum_is_late': 'installments_late_payment_count',
        'installments_mean_AMT_PAYMENT': 'installments_avg_payment_amount',
        'installments_mean_AMT_INSTALMENT': 'installments_avg_instalment_amount',
        'installments_std_payment_ratio': 'installments_payment_consistency',
        'installments_mean_days_late': 'installments_avg_days_late',
        'installments_sum_AMT_PAYMENT': 'installments_total_paid'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in installments_agg.columns:
            installments_agg = installments_agg.rename(columns={old_name: new_name})
    
    # Calculate payment consistency (inverse of standard deviation - higher is more consistent)
    if 'installments_payment_consistency' in installments_agg.columns:
        # Convert std to consistency score (lower std = higher consistency)
        installments_agg['installments_payment_consistency'] = (
            1 / (1 + installments_agg['installments_payment_consistency'])
        ).fillna(1)  # Perfect consistency if no variation
    else:
        installments_agg['installments_payment_consistency'] = 1
    
    # Fill missing values
    numeric_columns = installments_agg.select_dtypes(include=[np.number]).columns
    installments_agg[numeric_columns] = installments_agg[numeric_columns].fillna(0)
    
    return installments_agg


def credit_card_features(cc_balance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from credit card balance data.
    
    Args:
        cc_balance_df: DataFrame containing credit card balance data
        
    Returns:
        DataFrame with aggregated credit card features by SK_ID_CURR
    """
    if cc_balance_df.empty:
        # Return empty DataFrame with expected columns if no credit card data
        return pd.DataFrame(columns=[
            'SK_ID_CURR', 'cc_balance_count', 'cc_avg_balance_utilization',
            'cc_payment_behavior_score', 'cc_avg_balance', 'cc_avg_credit_limit',
            'cc_avg_payment_amount', 'cc_total_drawings', 'cc_avg_drawings_atm',
            'cc_avg_drawings_pos', 'cc_payment_consistency', 'cc_avg_dpd',
            'cc_max_dpd', 'cc_active_months'
        ])
    
    # Calculate balance utilization and payment behavior metrics
    cc_balance_df = cc_balance_df.copy()
    
    # Calculate balance utilization ratio (balance / credit limit)
    if 'AMT_BALANCE' in cc_balance_df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in cc_balance_df.columns:
        cc_balance_df['balance_utilization'] = (
            cc_balance_df['AMT_BALANCE'] / 
            cc_balance_df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        ).fillna(0)
    else:
        cc_balance_df['balance_utilization'] = 0
    
    # Calculate payment ratio (payment / minimum payment)
    if 'AMT_PAYMENT_CURRENT' in cc_balance_df.columns and 'AMT_INST_MIN_REGULARITY' in cc_balance_df.columns:
        cc_balance_df['payment_ratio'] = (
            cc_balance_df['AMT_PAYMENT_CURRENT'] / 
            cc_balance_df['AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
        ).fillna(0)
    else:
        cc_balance_df['payment_ratio'] = 0
    
    # Calculate total drawings (ATM + POS + Other)
    drawing_columns = ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT']
    available_drawing_columns = [col for col in drawing_columns if col in cc_balance_df.columns]
    
    if available_drawing_columns:
        cc_balance_df['total_drawings'] = cc_balance_df[available_drawing_columns].fillna(0).sum(axis=1)
    else:
        cc_balance_df['total_drawings'] = 0
    
    # Calculate spending patterns (drawings / credit limit)
    if 'AMT_CREDIT_LIMIT_ACTUAL' in cc_balance_df.columns:
        cc_balance_df['spending_ratio'] = (
            cc_balance_df['total_drawings'] / 
            cc_balance_df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        ).fillna(0)
    else:
        cc_balance_df['spending_ratio'] = 0
    
    print(f"Processing credit card data with shape: {cc_balance_df.shape}")
    print(f"Available columns: {list(cc_balance_df.columns)}")
    
    # Define aggregations based on available columns
    agg_dict = {
        'MONTHS_BALANCE': 'count',  # Total number of months with balance data
    }
    
    # Add aggregations for columns that exist
    if 'AMT_BALANCE' in cc_balance_df.columns:
        agg_dict['AMT_BALANCE'] = ['mean', 'max', 'std']
    if 'AMT_CREDIT_LIMIT_ACTUAL' in cc_balance_df.columns:
        agg_dict['AMT_CREDIT_LIMIT_ACTUAL'] = ['mean', 'max']
    if 'AMT_PAYMENT_CURRENT' in cc_balance_df.columns:
        agg_dict['AMT_PAYMENT_CURRENT'] = ['mean', 'sum', 'std']
    if 'AMT_PAYMENT_TOTAL_CURRENT' in cc_balance_df.columns:
        agg_dict['AMT_PAYMENT_TOTAL_CURRENT'] = ['mean', 'sum']
    if 'AMT_DRAWINGS_ATM_CURRENT' in cc_balance_df.columns:
        agg_dict['AMT_DRAWINGS_ATM_CURRENT'] = ['mean', 'sum']
    if 'AMT_DRAWINGS_POS_CURRENT' in cc_balance_df.columns:
        agg_dict['AMT_DRAWINGS_POS_CURRENT'] = ['mean', 'sum']
    if 'AMT_DRAWINGS_OTHER_CURRENT' in cc_balance_df.columns:
        agg_dict['AMT_DRAWINGS_OTHER_CURRENT'] = ['mean', 'sum']
    if 'CNT_DRAWINGS_ATM_CURRENT' in cc_balance_df.columns:
        agg_dict['CNT_DRAWINGS_ATM_CURRENT'] = ['mean', 'sum']
    if 'CNT_DRAWINGS_POS_CURRENT' in cc_balance_df.columns:
        agg_dict['CNT_DRAWINGS_POS_CURRENT'] = ['mean', 'sum']
    if 'SK_DPD' in cc_balance_df.columns:
        agg_dict['SK_DPD'] = ['mean', 'max', 'sum']
    if 'SK_DPD_DEF' in cc_balance_df.columns:
        agg_dict['SK_DPD_DEF'] = ['mean', 'max', 'sum']
    if 'balance_utilization' in cc_balance_df.columns:
        agg_dict['balance_utilization'] = ['mean', 'max', 'std']
    if 'payment_ratio' in cc_balance_df.columns:
        agg_dict['payment_ratio'] = ['mean', 'std']
    if 'total_drawings' in cc_balance_df.columns:
        agg_dict['total_drawings'] = ['mean', 'sum']
    if 'spending_ratio' in cc_balance_df.columns:
        agg_dict['spending_ratio'] = ['mean', 'max']
    
    print(f"Aggregating credit card data with columns: {list(agg_dict.keys())}")
    
    try:
        cc_agg = cc_balance_df.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    except Exception as e:
        print(f"Error during credit card aggregation: {e}")
        print(f"Available columns: {list(cc_balance_df.columns)}")
        raise
    
    # Flatten column names
    cc_agg.columns = ['SK_ID_CURR'] + [
        f'cc_{col[1]}_{col[0]}' if col[1] != '' else f'cc_{col[0]}'
        for col in cc_agg.columns[1:]
    ]
    
    # Rename key columns to match expected names
    column_mapping = {
        'cc_count_MONTHS_BALANCE': 'cc_active_months',
        'cc_mean_balance_utilization': 'cc_avg_balance_utilization',
        'cc_mean_AMT_BALANCE': 'cc_avg_balance',
        'cc_mean_AMT_CREDIT_LIMIT_ACTUAL': 'cc_avg_credit_limit',
        'cc_mean_AMT_PAYMENT_CURRENT': 'cc_avg_payment_amount',
        'cc_sum_total_drawings': 'cc_total_drawings',
        'cc_mean_AMT_DRAWINGS_ATM_CURRENT': 'cc_avg_drawings_atm',
        'cc_mean_AMT_DRAWINGS_POS_CURRENT': 'cc_avg_drawings_pos',
        'cc_std_payment_ratio': 'cc_payment_consistency',
        'cc_mean_SK_DPD': 'cc_avg_dpd',
        'cc_max_SK_DPD': 'cc_max_dpd'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in cc_agg.columns:
            cc_agg = cc_agg.rename(columns={old_name: new_name})
    
    # Calculate payment behavior score based on payment consistency and DPD
    # Higher score means better payment behavior
    payment_consistency = 1.0
    if 'cc_payment_consistency' in cc_agg.columns:
        # Convert payment ratio std to consistency score (lower std = higher consistency)
        payment_consistency = (
            1 / (1 + cc_agg['cc_payment_consistency'])
        ).fillna(1)  # Perfect consistency if no variation
    
    dpd_penalty = 1.0
    if 'cc_avg_dpd' in cc_agg.columns:
        # Penalize high days past due (lower score for higher DPD)
        dpd_penalty = 1 / (1 + cc_agg['cc_avg_dpd'])
    
    # Combine payment consistency and DPD penalty for overall payment behavior score
    cc_agg['cc_payment_behavior_score'] = payment_consistency * dpd_penalty
    
    # Ensure payment consistency column exists with proper values
    if 'cc_payment_consistency' not in cc_agg.columns:
        cc_agg['cc_payment_consistency'] = 1.0
    else:
        cc_agg['cc_payment_consistency'] = payment_consistency
    
    # Handle contract status if available
    if 'NAME_CONTRACT_STATUS' in cc_balance_df.columns:
        # Count active vs closed contracts
        status_counts = cc_balance_df.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
        
        if 'Active' in status_counts.columns:
            cc_agg = cc_agg.merge(
                status_counts[['Active']].rename(columns={'Active': 'cc_active_contracts'}),
                on='SK_ID_CURR', how='left'
            )
        else:
            cc_agg['cc_active_contracts'] = 0
            
        if 'Completed' in status_counts.columns:
            cc_agg = cc_agg.merge(
                status_counts[['Completed']].rename(columns={'Completed': 'cc_completed_contracts'}),
                on='SK_ID_CURR', how='left'
            )
        else:
            cc_agg['cc_completed_contracts'] = 0
    else:
        cc_agg['cc_active_contracts'] = 0
        cc_agg['cc_completed_contracts'] = 0
    
    # Ensure all expected columns exist with default values
    expected_columns = [
        'cc_balance_count', 'cc_avg_balance_utilization', 'cc_payment_behavior_score',
        'cc_avg_balance', 'cc_avg_credit_limit', 'cc_avg_payment_amount',
        'cc_total_drawings', 'cc_avg_drawings_atm', 'cc_avg_drawings_pos',
        'cc_payment_consistency', 'cc_avg_dpd', 'cc_max_dpd', 'cc_active_months'
    ]
    
    for col in expected_columns:
        if col not in cc_agg.columns:
            if col == 'cc_balance_count':
                cc_agg[col] = cc_agg.get('cc_active_months', 0)
            elif col in ['cc_payment_behavior_score', 'cc_payment_consistency']:
                cc_agg[col] = 1.0
            else:
                cc_agg[col] = 0
    
    # Fill missing values
    numeric_columns = cc_agg.select_dtypes(include=[np.number]).columns
    cc_agg[numeric_columns] = cc_agg[numeric_columns].fillna(0)
    
    return cc_agg


def merge_all_data(application_df: pd.DataFrame, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Master function to orchestrate the entire feature engineering process.
    Merges all engineered features with the main application dataset.
    
    Args:
        application_df: Main application DataFrame with SK_ID_CURR and target variable
        data_dict: Dictionary containing all auxiliary datasets
        
    Returns:
        DataFrame with all features merged by SK_ID_CURR
    """
    if application_df.empty:
        raise ValueError("Application DataFrame cannot be empty")
    
    if 'SK_ID_CURR' not in application_df.columns:
        raise ValueError("Application DataFrame must contain SK_ID_CURR column")
    
    # Start with the main application data
    merged_df = application_df.copy()
    
    print(f"Starting with {len(merged_df)} application records")
    
    # Process bureau data if available
    if 'bureau' in data_dict and not data_dict['bureau'].empty:
        print("Processing bureau features...")
        try:
            bureau_balance = data_dict.get('bureau_balance', pd.DataFrame())
            bureau_features_df = bureau_features(data_dict['bureau'], bureau_balance)
            
            if not bureau_features_df.empty:
                merged_df = merged_df.merge(bureau_features_df, on='SK_ID_CURR', how='left')
                print(f"Added {len(bureau_features_df.columns) - 1} bureau features")
            else:
                print("No bureau features generated (empty result)")
        except Exception as e:
            print(f"Error processing bureau features: {e}")
            raise Exception(f"Failed to process bureau features: {e}")
    else:
        print("No bureau data available, skipping bureau features")
    
    # Process previous applications data if available
    if 'previous_application' in data_dict and not data_dict['previous_application'].empty:
        print("Processing previous application features...")
        try:
            prev_app_features_df = previous_application_features(data_dict['previous_application'])
            
            if not prev_app_features_df.empty:
                merged_df = merged_df.merge(prev_app_features_df, on='SK_ID_CURR', how='left')
                print(f"Added {len(prev_app_features_df.columns) - 1} previous application features")
            else:
                print("No previous application features generated (empty result)")
        except Exception as e:
            print(f"Error processing previous application features: {e}")
            print("Adding default previous application features with zero values")
            # Add default previous application feature columns with zero values
            default_prev_app_cols = [
                'prev_app_count', 'prev_app_avg_credit', 'prev_app_approval_rate',
                'prev_app_avg_annuity', 'prev_app_avg_goods_price', 'prev_app_refused_count',
                'prev_app_approved_count', 'prev_app_avg_days_decision'
            ]
            for col in default_prev_app_cols:
                merged_df[col] = 0
    else:
        print("No previous application data available, skipping previous application features")
        # Add default previous application feature columns with zero values
        default_prev_app_cols = [
            'prev_app_count', 'prev_app_avg_credit', 'prev_app_approval_rate',
            'prev_app_avg_annuity', 'prev_app_avg_goods_price', 'prev_app_refused_count',
            'prev_app_approved_count', 'prev_app_avg_days_decision'
        ]
        for col in default_prev_app_cols:
            merged_df[col] = 0
    
    # Process installments data if available
    if 'installments_payments' in data_dict and not data_dict['installments_payments'].empty:
        print("Processing installments features...")
        try:
            installments_features_df = installments_features(data_dict['installments_payments'])
            
            if not installments_features_df.empty:
                merged_df = merged_df.merge(installments_features_df, on='SK_ID_CURR', how='left')
                print(f"Added {len(installments_features_df.columns) - 1} installments features")
            else:
                print("No installments features generated (empty result)")
        except Exception as e:
            print(f"Error processing installments features: {e}")
            print("Adding default installments features with zero values")
            # Add default installments feature columns with zero values
            default_installments_cols = [
                'installments_count', 'installments_payment_ratio', 'installments_late_payment_count',
                'installments_avg_payment_amount', 'installments_avg_instalment_amount',
                'installments_payment_consistency', 'installments_avg_days_late', 'installments_total_paid'
            ]
            for col in default_installments_cols:
                merged_df[col] = 0
    else:
        print("No installments data available, skipping installments features")
        # Add default installments feature columns with zero values
        default_installments_cols = [
            'installments_count', 'installments_payment_ratio', 'installments_late_payment_count',
            'installments_avg_payment_amount', 'installments_avg_instalment_amount',
            'installments_payment_consistency', 'installments_avg_days_late', 'installments_total_paid'
        ]
        for col in default_installments_cols:
            merged_df[col] = 0
    
    # Process credit card balance data if available
    if 'credit_card_balance' in data_dict and not data_dict['credit_card_balance'].empty:
        print("Processing credit card features...")
        try:
            cc_features_df = credit_card_features(data_dict['credit_card_balance'])
            
            if not cc_features_df.empty:
                merged_df = merged_df.merge(cc_features_df, on='SK_ID_CURR', how='left')
                print(f"Added {len(cc_features_df.columns) - 1} credit card features")
            else:
                print("No credit card features generated (empty result)")
        except Exception as e:
            print(f"Error processing credit card features: {e}")
            print("Adding default credit card features with zero values")
            # Add default credit card feature columns with zero values
            default_cc_cols = [
                'cc_balance_count', 'cc_avg_balance_utilization', 'cc_payment_behavior_score',
                'cc_avg_balance', 'cc_avg_credit_limit', 'cc_avg_payment_amount',
                'cc_total_drawings', 'cc_avg_drawings_atm', 'cc_avg_drawings_pos',
                'cc_payment_consistency', 'cc_avg_dpd', 'cc_max_dpd', 'cc_active_months'
            ]
            for col in default_cc_cols:
                if col in ['cc_payment_behavior_score', 'cc_payment_consistency']:
                    merged_df[col] = 1.0  # Default to perfect behavior/consistency
                else:
                    merged_df[col] = 0
    else:
        print("No credit card data available, skipping credit card features")
        # Add default credit card feature columns with zero values
        default_cc_cols = [
            'cc_balance_count', 'cc_avg_balance_utilization', 'cc_payment_behavior_score',
            'cc_avg_balance', 'cc_avg_credit_limit', 'cc_avg_payment_amount',
            'cc_total_drawings', 'cc_avg_drawings_atm', 'cc_avg_drawings_pos',
            'cc_payment_consistency', 'cc_avg_dpd', 'cc_max_dpd', 'cc_active_months'
        ]
        for col in default_cc_cols:
            if col in ['cc_payment_behavior_score', 'cc_payment_consistency']:
                merged_df[col] = 1.0  # Default to perfect behavior/consistency
            else:
                merged_df[col] = 0
    
    # Fill any remaining missing values in engineered features
    # Get all feature columns (exclude SK_ID_CURR and TARGET if present)
    feature_columns = [col for col in merged_df.columns 
                      if col not in ['SK_ID_CURR', 'TARGET'] and merged_df[col].dtype in ['int64', 'float64']]
    
    # Fill missing values with 0 for engineered features (indicates no credit history)
    merged_df[feature_columns] = merged_df[feature_columns].fillna(0)
    
    print(f"Feature engineering complete. Final dataset shape: {merged_df.shape}")
    print(f"Total features added: {len(feature_columns)}")
    
    return merged_df