"""
Model saving and loading utilities for Home Credit Default Risk project.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from datetime import datetime


def save_model_complete(model, feature_names, model_config, performance_metrics, 
                       model_name="home_credit_model"):
    """
    Save the complete model package including model, features, config, and metrics.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names used in training
        model_config: Model configuration parameters
        performance_metrics: Dictionary of performance metrics
        model_name: Base name for saved files
    
    Returns:
        Dict with paths of saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. Save the trained model (pickle - most reliable)
    model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    saved_files['model'] = model_path
    
    # 2. Save feature names
    features_path = os.path.join(models_dir, f"{model_name}_features_{timestamp}.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    saved_files['features'] = features_path
    
    # 3. Save model configuration
    config_path = os.path.join(models_dir, f"{model_name}_config_{timestamp}.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    saved_files['config'] = config_path
    
    # 4. Save performance metrics
    metrics_path = os.path.join(models_dir, f"{model_name}_metrics_{timestamp}.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(performance_metrics, f)
    saved_files['metrics'] = metrics_path
    
    # 5. Save model info summary
    model_info = {
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'roc_auc': performance_metrics.get('roc_auc', 'N/A'),
        'feature_names': feature_names,
        'model_config': model_config
    }
    
    info_path = os.path.join(models_dir, f"{model_name}_info_{timestamp}.json")
    import json
    with open(info_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_safe_info = {k: convert_numpy(v) for k, v in model_info.items()}
        json.dump(json_safe_info, f, indent=2)
    saved_files['info'] = info_path
    
    print("✅ Model package saved successfully!")
    print(f"📁 Files saved:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    return saved_files


def load_model_complete(model_path, features_path=None, config_path=None, metrics_path=None):
    """
    Load the complete model package.
    
    Args:
        model_path: Path to the saved model file
        features_path: Path to the saved features file (optional)
        config_path: Path to the saved config file (optional)
        metrics_path: Path to the saved metrics file (optional)
    
    Returns:
        Dictionary containing loaded model and metadata
    """
    result = {}
    
    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        result['model'] = pickle.load(f)
    
    # Load features if provided
    if features_path and os.path.exists(features_path):
        print(f"Loading features from {features_path}...")
        with open(features_path, 'rb') as f:
            result['features'] = pickle.load(f)
    
    # Load config if provided
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'rb') as f:
            result['config'] = pickle.load(f)
    
    # Load metrics if provided
    if metrics_path and os.path.exists(metrics_path):
        print(f"Loading metrics from {metrics_path}...")
        with open(metrics_path, 'rb') as f:
            result['metrics'] = pickle.load(f)
    
    print("✅ Model package loaded successfully!")
    return result


def save_model_simple(model, filename="home_credit_model.pkl"):
    """
    Simple model saving function.
    
    Args:
        model: Trained model to save
        filename: Name of the file to save to
    """
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # If filename doesn't include path, save to models directory
    if not os.path.dirname(filename):
        filepath = os.path.join(models_dir, filename)
    else:
        filepath = filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {filepath}")


def load_model_simple(filename="home_credit_model.pkl"):
    """
    Simple model loading function.
    
    Args:
        filename: Name of the file to load from
    
    Returns:
        Loaded model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {filename}")
    return model


# For Colab - easy download function
def prepare_for_download(model, X_train, performance_metrics, base_name="home_credit"):
    """
    Prepare model files for download from Colab.
    
    Args:
        model: Trained model
        X_train: Training features (for feature names)
        performance_metrics: Model performance metrics
        base_name: Base name for files
    
    Returns:
        List of files ready for download
    """
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files_created = []
    
    # Save model
    model_file = os.path.join(models_dir, f"{base_name}_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    files_created.append(model_file)
    
    # Save feature names
    features_file = os.path.join(models_dir, f"{base_name}_features.pkl")
    with open(features_file, 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    files_created.append(features_file)
    
    # Save performance summary as CSV for easy viewing
    metrics_file = os.path.join(models_dir, f"{base_name}_performance.csv")
    metrics_df = pd.DataFrame([{
        'ROC_AUC': performance_metrics.get('roc_auc', 0),
        'Precision': performance_metrics.get('precision', 0),
        'Recall': performance_metrics.get('recall', 0),
        'F1_Score': performance_metrics.get('f1_score', 0),
        'CV_Mean': performance_metrics.get('cv_mean', 0),
        'CV_Std': performance_metrics.get('cv_std', 0)
    }])
    metrics_df.to_csv(metrics_file, index=False)
    files_created.append(metrics_file)
    
    # Save feature importance as CSV
    if 'feature_importance' in performance_metrics:
        importance_file = os.path.join(models_dir, f"{base_name}_feature_importance.csv")
        performance_metrics['feature_importance'].to_csv(importance_file, index=False)
        files_created.append(importance_file)
    
    print("✅ Files prepared for download:")
    for file in files_created:
        print(f"   📄 {file}")
    
    return files_created


if __name__ == "__main__":
    print("Model utilities loaded successfully!")
    print("Available functions:")
    print("  - save_model_complete(): Save model with all metadata")
    print("  - load_model_complete(): Load complete model package")
    print("  - save_model_simple(): Simple model saving")
    print("  - load_model_simple(): Simple model loading")
    print("  - prepare_for_download(): Prepare files for Colab download")