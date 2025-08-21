"""
Machine learning model module for Home Credit Default Risk prediction.
Handles model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


def train_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier for loan default prediction.
    Fixed to handle categorical features consistently.
    
    Args:
        X (pd.DataFrame): Feature matrix with training data
        y (pd.Series): Target variable (0 = no default, 1 = default)
    
    Returns:
        lgb.LGBMClassifier: Trained LightGBM model
    
    Raises:
        ValueError: If input data is invalid or empty
        Exception: If model training fails
    """
    if X.empty or y.empty:
        raise ValueError("Training data cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("Feature matrix and target must have same length")
    
    if y.nunique() < 2:
        raise ValueError("Target variable must have at least 2 classes")
    
    try:
        # Configure LightGBM parameters for imbalanced classification
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'is_unbalance': True  # Handle imbalanced dataset
        }
        
        # Initialize and train the model
        model = lgb.LGBMClassifier(**lgb_params)
        
        # Process data consistently - convert all to numeric
        X_processed = X.copy()
        
        # Convert any categorical columns to numeric
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                X_processed[col] = pd.Categorical(X_processed[col]).codes
            
            # Ensure all columns are numeric
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        
        # Fill any NaN values that might have been created
        X_processed = X_processed.fillna(0)
        
        # Train without specifying categorical features to avoid mismatch issues
        model.fit(
            X_processed, y,
            eval_set=[(X_processed, y)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        return model
        
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")


def evaluate_model(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the trained model and return comprehensive performance metrics.
    
    Args:
        model (lgb.LGBMClassifier): Trained LightGBM model
        X (pd.DataFrame): Feature matrix for evaluation
        y (pd.Series): True target values
    
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics and visualization data
    
    Raises:
        ValueError: If input data is invalid
        Exception: If model evaluation fails
    """
    if X.empty or y.empty:
        raise ValueError("Evaluation data cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("Feature matrix and target must have same length")
    
    try:
        # Process data consistently - same as training
        X_processed = X.copy()
        
        # Convert any categorical columns to numeric
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                X_processed[col] = pd.Categorical(X_processed[col]).codes
            
            # Ensure all columns are numeric
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        
        # Fill any NaN values that might have been created
        X_processed = X_processed.fillna(0)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        y_pred = model.predict(X_processed)
        
        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Generate ROC curve data
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        
        # Get classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Cross-validation with stratified k-fold to prevent data leakage
        cv_scores = cross_val_score(
            model, X_processed, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'roc_curve_data': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'classification_report': class_report,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score']
        }
        
    except Exception as e:
        raise Exception(f"Model evaluation failed: {str(e)}")


def predict(model: lgb.LGBMClassifier, data: pd.DataFrame) -> np.ndarray:
    """
    Make probability predictions for default risk assessment.
    
    Args:
        model (lgb.LGBMClassifier): Trained LightGBM model
        data (pd.DataFrame): Feature data for prediction
    
    Returns:
        np.ndarray: Array of probability scores between 0 and 1 for default risk
    
    Raises:
        ValueError: If input data is invalid
        Exception: If prediction fails
    """
    if data.empty:
        raise ValueError("Prediction data cannot be empty")
    
    try:
        # Process data consistently - same as training
        data_processed = data.copy()
        
        # Convert any categorical columns to numeric
        for col in data_processed.columns:
            if data_processed[col].dtype == 'object' or data_processed[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                data_processed[col] = pd.Categorical(data_processed[col]).codes
            
            # Ensure all columns are numeric
            data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
        
        # Fill any NaN values that might have been created
        data_processed = data_processed.fillna(0)
        
        # Handle feature alignment - ensure prediction data has same features as training
        model_features = model.feature_name_
        
        # Check for missing columns and add them with default values
        missing_cols = set(model_features) - set(data_processed.columns)
        if missing_cols:
            for col in missing_cols:
                # Add missing columns with appropriate default values
                if col.endswith('_count') or col.endswith('_num'):
                    data_processed[col] = 0
                elif col.endswith('_rate') or col.endswith('_ratio'):
                    data_processed[col] = 0.0
                else:
                    data_processed[col] = 0
        
        # Ensure column order matches training data
        data_aligned = data_processed[model_features]
        
        # Make probability predictions
        probabilities = model.predict_proba(data_aligned)[:, 1]
        
        # Ensure probabilities are between 0 and 1
        probabilities = np.clip(probabilities, 0, 1)
        
        return probabilities
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")