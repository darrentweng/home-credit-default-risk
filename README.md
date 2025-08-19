# Home Credit Default Risk

## Overview

This project aims to predict the probability of a loan applicant defaulting on their loan. It is a classic machine learning classification problem that is fundamental to the finance and banking industry. The solution involves an end-to-end data science pipeline, from complex data integration and feature engineering to model training and interactive visualization. The final product is a Streamlit application that demonstrates the model's performance and allows for interactive analysis.

## Problem Statement

Given a set of applicant data from various sources (loan applications, credit bureau data, previous loan историю, etc.), build a model to predict if a new applicant will repay their loan on time or default. The model's predictions will be used to assess credit risk and inform lending decisions.

## Project Structure

The project is structured for modularity and clarity. Each component of the data science pipeline is contained within a separate file.

```
.
├── README.md
├── app.py
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── feature_engineering.py
    ├── model.py
    └── utils.py
```

## Getting Started

### 1. Data Source

The data is from the **Home Credit Default Risk** competition on Kaggle. You must download the following files from the competition's data page and place them in a `data/` directory at the root of the project:

- `application_train.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

### 2. Dependencies

The project requires several popular Python libraries. Install them using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```
streamlit
pandas
numpy
scikit-learn
lightgbm
shap
```

## Implementation

### `src/data_processing.py`
This script handles the loading and initial cleaning of the raw data files.

- `load_all_data()`: Loads all the CSV files into a dictionary of Pandas DataFrames.
- `clean_data()`: Performs initial data cleaning, such as handling missing values and correcting data types for each DataFrame.

### `src/feature_engineering.py`
This is the most critical part of the project. This script contains the logic for creating new features from the raw data.

- `bureau_features(df)`: Creates aggregated features from `bureau.csv` and `bureau_balance.csv` (e.g., number of past loans, average debt).
- `previous_application_features(df)`: Creates features from `previous_application.csv` (e.g., number of previous applications, average loan amount).
- `installments_features(df)`: Aggregates payment history from `installments_payments.csv` to create features like payment-to-loan-amount ratio.
- `merge_all_data()`: A master function that orchestrates the entire feature engineering pipeline, merging all the new features into the main `application_train.csv` DataFrame.

### `src/model.py`
This script handles all the machine learning modeling aspects.

- `train_model(X, y)`: Trains a LightGBM Classifier model. It includes logic for handling categorical features and can be configured with cross-validation.
- `evaluate_model(model, X, y)`: Evaluates the trained model and returns key performance metrics, specifically ROC-AUC.
- `predict(model, data)`: A function to make predictions on new, unseen data.

### `app.py`
The main Streamlit application script. It ties all the components together to create an interactive demo.

- **Data Loading**: The app first loads and processes the data using the functions from `src/`.
- **Model Training**: It then trains the LightGBM model. To avoid retraining on every run, the model can be cached using Streamlit's `@st.cache_resource` decorator.
- **Dashboard**: The app displays key insights and model performance metrics, such as ROC-AUC, a confusion matrix, and a ROC curve plot.
- **Feature Importance**: It uses the `shap` library to show which features were most influential in the model's predictions, providing valuable model interpretability.
- **Interactive Prediction**: The app provides a sidebar where users can input hypothetical applicant data (e.g., income, loan amount) and get a real-time risk prediction from the trained model.

## How to Run the App

1. Clone the repository.
2. Download the data files from Kaggle and place them in a `data/` folder.
3. Install the dependencies: `pip install -r requirements.txt`.
4. Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```