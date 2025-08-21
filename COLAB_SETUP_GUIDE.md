# Google Colab Setup Guide - Home Credit Model

## Quick Start (5 Steps)

### 1. Upload Files to Colab
Upload these files/folders to your Colab environment:
- `src/` folder (with all your .py files)
- `data/` folder (with Kaggle CSV files)

### 2. Install Packages
```python
!pip install lightgbm pandas numpy scikit-learn matplotlib seaborn plotly shap
```

### 3. Import Your Modules
```python
import sys
sys.path.append('src')

from data_processing import load_all_data, clean_data
from feature_engineering import merge_all_data
from model import train_model, evaluate_model, predict
from utils import get_risk_category, format_currency
```

### 4. Run the Pipeline
```python
# Load data
data_dict = load_all_data('data')

# Clean data
cleaned_data = clean_data(data_dict)

# Engineer features
application_df = cleaned_data['application_train']
engineered_df = merge_all_data(application_df, cleaned_data)

# Prepare for modeling
X = engineered_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = engineered_df['TARGET']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = train_model(X_train, y_train)

# Evaluate
results = evaluate_model(model, X_test, y_test)
print(f"ROC-AUC: {results['roc_auc']:.4f}")
```

### 5. Make Predictions
```python
# Get predictions
probabilities = predict(model, X_test)

# Analyze results
import pandas as pd
results_df = pd.DataFrame({
    'Probability': probabilities,
    'Risk': [get_risk_category(p) for p in probabilities],
    'Actual': y_test
})

print(results_df['Risk'].value_counts())
```

## Detailed Cell-by-Cell Instructions

### Cell 1: Setup
```python
# Install packages
!pip install lightgbm pandas numpy scikit-learn matplotlib seaborn plotly shap

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
print("✅ Setup complete!")
```

### Cell 2: Import Modules
```python
# Import your custom modules
from data_processing import load_all_data, clean_data
from feature_engineering import merge_all_data
from model import train_model, evaluate_model, predict
from utils import get_risk_category, format_currency, validate_data_schema

print("✅ All modules imported!")
```

### Cell 3: Load Data
```python
# Load all CSV files
print("Loading data...")
data_dict = load_all_data('data')

print("Data loaded:")
for name, df in data_dict.items():
    print(f"  {name}: {df.shape}")
```

### Cell 4: Clean Data
```python
# Clean all datasets
print("Cleaning data...")
cleaned_data = clean_data(data_dict)

print("Data cleaned:")
for name, df in cleaned_data.items():
    missing = df.isnull().sum().sum()
    print(f"  {name}: {df.shape}, Missing: {missing:,}")
```

### Cell 5: Feature Engineering
```python
# Create engineered features
print("Engineering features...")
application_df = cleaned_data['application_train']
engineered_df = merge_all_data(application_df, cleaned_data)

print(f"Final dataset: {engineered_df.shape}")
print(f"Default rate: {engineered_df['TARGET'].mean():.2%}")
```

### Cell 6: Prepare Data
```python
# Separate features and target
X = engineered_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = engineered_df['TARGET']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features: {X.shape[1]}")
print(f"Training: {len(X_train):,} samples")
print(f"Testing: {len(X_test):,} samples")
```

### Cell 7: Train Model
```python
# Train the model (this takes time!)
print("Training model... (this may take 5-10 minutes)")
model = train_model(X_train, y_train)
print("✅ Training complete!")
```

### Cell 8: Evaluate Model
```python
# Evaluate performance
print("Evaluating model...")
results = evaluate_model(model, X_test, y_test)

print("Performance Results:")
print(f"  ROC-AUC: {results['roc_auc']:.4f}")
print(f"  Precision: {results['precision']:.4f}")
print(f"  Recall: {results['recall']:.4f}")
print(f"  F1-Score: {results['f1_score']:.4f}")

if results['roc_auc'] > 0.7:
    print("✅ Excellent performance!")
elif results['roc_auc'] > 0.6:
    print("✅ Good performance!")
else:
    print("⚠️ Consider model improvements")
```

### Cell 9: Make Predictions
```python
# Generate predictions
probabilities = predict(model, X_test)

# Create results dataframe
results_df = pd.DataFrame({
    'SK_ID_CURR': engineered_df.loc[X_test.index, 'SK_ID_CURR'],
    'Actual': y_test,
    'Probability': probabilities,
    'Risk': [get_risk_category(p) for p in probabilities]
})

# Show risk distribution
print("Risk Distribution:")
risk_counts = results_df['Risk'].value_counts()
for risk, count in risk_counts.items():
    pct = count / len(results_df) * 100
    print(f"  {risk}: {count:,} ({pct:.1f}%)")
```

### Cell 10: Analyze Results
```python
# Show sample predictions
print("Sample High Risk Cases:")
high_risk = results_df[results_df['Risk'] == 'High'].head(5)
for _, row in high_risk.iterrows():
    correct = "✅" if (row['Probability'] > 0.5) == row['Actual'] else "❌"
    print(f"  ID: {row['SK_ID_CURR']} | Prob: {row['Probability']:.3f} | Actual: {row['Actual']} {correct}")

print("\nSample Low Risk Cases:")
low_risk = results_df[results_df['Risk'] == 'Low'].head(5)
for _, row in low_risk.iterrows():
    correct = "✅" if (row['Probability'] > 0.5) == row['Actual'] else "❌"
    print(f"  ID: {row['SK_ID_CURR']} | Prob: {row['Probability']:.3f} | Actual: {row['Actual']} {correct}")
```

### Cell 11: Feature Importance
```python
# Show top features
print("Top 15 Important Features:")
feature_importance = results['feature_importance']
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Cell 12: Save Results
```python
# Save model and results
import pickle

# Save trained model
with open('home_credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save predictions
results_df.to_csv('predictions.csv', index=False)

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("✅ Files saved:")
print("  📄 home_credit_model.pkl")
print("  📄 predictions.csv") 
print("  📄 feature_names.pkl")
print("\nDownload these files to use in your Streamlit app!")
```

## Expected Results

With real Kaggle data, you should see:
- **ROC-AUC**: 0.70-0.78 (good to excellent)
- **Training time**: 5-15 minutes depending on data size
- **Risk distribution**: ~70% Low, ~25% Medium, ~5% High
- **Top features**: EXT_SOURCE_*, AMT_CREDIT, AMT_INCOME_TOTAL, bureau features

## Troubleshooting

### Common Issues:
1. **Import errors**: Make sure `src/` folder is uploaded with all .py files
2. **Data not found**: Ensure `data/` folder contains all 6 CSV files
3. **Memory errors**: Use smaller sample with `df.sample(n=10000)` 
4. **Slow training**: Reduce `n_estimators` in LIGHTGBM_PARAMS

### File Requirements:
- `src/data_processing.py`
- `src/feature_engineering.py` 
- `src/model.py`
- `src/utils.py`
- `data/application_train.csv`
- `data/bureau.csv`
- `data/bureau_balance.csv`
- `data/previous_application.csv`
- `data/installments_payments.csv`
- `data/credit_card_balance.csv`

## Next Steps

Once you get good results:
1. Download the saved model files
2. Build your Streamlit app using these files
3. Deploy to Streamlit Cloud or other platform

The model is now ready for production use!