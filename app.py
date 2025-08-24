"""
Streamlit web application for Home Credit Default Risk prediction.
Interactive interface for model analysis and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import shap
import warnings
from typing import Dict, List, Optional, Any

# Import our custom modules
from src.data_processing import load_all_data, clean_data
from src.feature_engineering import merge_all_data
from src.model import train_model, evaluate_model, predict
from src.utils import (
    get_risk_category, format_currency, format_percentage,
    calculate_shap_values, validate_data_schema, log_data_info,
    FEATURE_GROUPS
)
from model_utils import save_model_complete

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    """
    Load and process all data with caching for performance.
    This function is cached to avoid reloading data on every interaction.
    """
    try:
        with st.spinner("Loading data files..."):
            # Load all data
            data_dict = load_all_data('data')
            
            # Clean the data
            cleaned_data = clean_data(data_dict)
            
            # Feature engineering
            application_df = cleaned_data['application_train']
            merged_df = merge_all_data(application_df, cleaned_data)
            
            # Prepare features and target
            target_column = 'TARGET'
            feature_columns = [col for col in merged_df.columns 
                             if col not in ['SK_ID_CURR', target_column]]
            
            X = merged_df[feature_columns]
            y = merged_df[target_column]
            
            return X, y, merged_df, feature_columns
            
    except FileNotFoundError as e:
        st.error(f"❌ **Data files not found**: {str(e)}")
        st.error("Please ensure you have downloaded the required CSV files from Kaggle and placed them in the 'data/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error loading data**: {str(e)}")
        st.stop()


@st.cache_resource
def train_cached_model(X, y):
    """
    Train the model with caching to avoid retraining on every run.
    This function is cached based on the input data hash.
    """
    try:
        with st.spinner("Training LightGBM model... This may take a few minutes."):
            model = train_model(X, y)
            return model
    except Exception as e:
        st.error(f"❌ **Error training model**: {str(e)}")
        st.stop()


def create_confusion_matrix_plot(cm):
    """Create an interactive confusion matrix visualization."""
    labels = ['No Default (0)', 'Default (1)']
    
    # Normalize confusion matrix for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=[[f'{cm[i][j]}<br>({cm_normalized[i][j]:.1%})' 
                         for j in range(len(cm[i]))] for i in range(len(cm))],
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=400,
        height=400
    )
    
    return fig


def create_roc_curve_plot(roc_data):
    """Create an interactive ROC curve visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=roc_data['fpr'],
        y=roc_data['tpr'],
        mode='lines',
        name='ROC Curve',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=500,
        height=400,
        showlegend=True
    )
    
    return fig


def create_feature_importance_plot(feature_importance_df, top_n=20):
    """Create feature importance visualization."""
    top_features = feature_importance_df.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_shap_summary_plot(shap_values, X, max_features=20):
    """Create SHAP summary plot using Plotly."""
    try:
        # Get mean absolute SHAP values for each feature
        mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(max_features)
        
        fig = go.Figure(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title=f'SHAP Feature Importance (Top {max_features})',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating SHAP plot: {str(e)}")
        return None


def render_sidebar_prediction(model, feature_columns):
    """Render the interactive prediction sidebar."""
    st.sidebar.markdown('<p class="sidebar-header">🔮 Interactive Risk Prediction</p>', 
                       unsafe_allow_html=True)
    
    st.sidebar.markdown("**Adjust the parameters below to see real-time default risk prediction:**")
    
    # Group inputs by category for better UX
    with st.sidebar.expander("💰 Financial Information", expanded=True):
        income = st.number_input("Annual Income ($)", 
                               min_value=0, max_value=1000000, value=150000, step=5000)
        credit_amount = st.number_input("Credit Amount ($)", 
                                      min_value=0, max_value=1000000, value=300000, step=10000)
        annuity = st.number_input("Loan Annuity ($)", 
                                min_value=0, max_value=100000, value=15000, step=1000)
        goods_price = st.number_input("Goods Price ($)", 
                                    min_value=0, max_value=1000000, value=280000, step=10000)
    
    with st.sidebar.expander("👤 Personal Information"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", 
                                ["Secondary / secondary special", "Higher education", 
                                 "Incomplete higher", "Lower secondary", "Academic degree"])
        family_status = st.selectbox("Family Status", 
                                   ["Married", "Single / not married", "Civil marriage", 
                                    "Separated", "Widow"])
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        days_employed = st.slider("Years Employed", min_value=0, max_value=40, value=5)
    
    with st.sidebar.expander("🏢 Employment & Income"):
        income_type = st.selectbox("Income Type", 
                                 ["Working", "State servant", "Commercial associate", 
                                  "Pensioner", "Unemployed", "Student", "Businessman"])
        occupation = st.selectbox("Occupation", 
                                ["Laborers", "Core staff", "Sales staff", "Managers", 
                                 "Drivers", "High skill tech staff", "Accountants", 
                                 "Medicine staff", "Security staff", "Cooking staff"])
    
    # Create prediction button
    if st.sidebar.button("🎯 Predict Default Risk", type="primary"):
        try:
            # Create a prediction DataFrame with all required features
            prediction_data = pd.DataFrame({col: [0] for col in feature_columns})
            
            # Set the user inputs (map to actual feature names if they exist)
            if 'AMT_INCOME_TOTAL' in feature_columns:
                prediction_data['AMT_INCOME_TOTAL'] = [income]
            if 'AMT_CREDIT' in feature_columns:
                prediction_data['AMT_CREDIT'] = [credit_amount]
            if 'AMT_ANNUITY' in feature_columns:
                prediction_data['AMT_ANNUITY'] = [annuity]
            if 'AMT_GOODS_PRICE' in feature_columns:
                prediction_data['AMT_GOODS_PRICE'] = [goods_price]
            if 'CNT_CHILDREN' in feature_columns:
                prediction_data['CNT_CHILDREN'] = [children]
            if 'DAYS_BIRTH' in feature_columns:
                prediction_data['DAYS_BIRTH'] = [-(age * 365)]  # Convert age to days (negative)
            if 'DAYS_EMPLOYED' in feature_columns:
                prediction_data['DAYS_EMPLOYED'] = [-(days_employed * 365)]  # Convert to days (negative)
            
            # Convert categorical inputs to numeric (simplified encoding)
            if 'CODE_GENDER' in feature_columns:
                prediction_data['CODE_GENDER'] = [1 if gender == "Female" else 0]
            
            # Make prediction
            risk_probability = predict(model, prediction_data)[0]
            risk_category = get_risk_category(risk_probability)
            
            # Display results with styling
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🎯 Prediction Results:**")
            
            # Risk probability with color coding
            risk_color_class = f"risk-{risk_category.lower()}"
            st.sidebar.markdown(f"""
            <div class="metric-card">
                <h3>Default Probability</h3>
                <h2 class="{risk_color_class}">{format_percentage(risk_probability)}</h2>
                <p><strong>Risk Level: <span class="{risk_color_class}">{risk_category}</span></strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk interpretation
            if risk_category == "Low":
                st.sidebar.success("✅ **Low Risk**: This applicant has a low probability of default. Recommended for loan approval.")
            elif risk_category == "Medium":
                st.sidebar.warning("⚠️ **Medium Risk**: This applicant requires additional review. Consider with caution.")
            else:
                st.sidebar.error("❌ **High Risk**: This applicant has a high probability of default. Not recommended for loan approval.")
                
        except Exception as e:
            st.sidebar.error(f"Error making prediction: {str(e)}")


def main():
    """Main application function."""
    # App header
    st.title("🏦 Home Credit Default Risk Prediction")
    st.markdown("""
    **A comprehensive machine learning solution for credit risk assessment**
    
    This application predicts the probability of loan default using advanced machine learning techniques
    and provides interactive insights into the decision-making process.
    """)
    
    # Load and process data
    try:
        X, y, merged_df, feature_columns = load_and_process_data()
        
        # Display data loading success
        st.success(f"✅ **Data loaded successfully!** {len(merged_df):,} records with {len(feature_columns)} features")
        
        # Train model
        model = train_cached_model(X, y)
        
        # Model evaluation
        with st.spinner("Evaluating model performance..."):
            evaluation_results = evaluate_model(model, X, y)
        
        # Save the trained model
        with st.spinner("Saving trained model..."):
            try:
                # Prepare model configuration
                model_config = {
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
                    'is_unbalance': True
                }
                
                # Save the complete model package
                saved_files = save_model_complete(
                    model=model,
                    feature_names=list(X.columns),
                    model_config=model_config,
                    performance_metrics=evaluation_results,
                    model_name="home_credit_model"
                )
                
                # Store saved files info in session state for UI display
                st.session_state.saved_model_files = saved_files
                st.session_state.model_saved_successfully = True
                
            except Exception as e:
                st.warning(f"⚠️ **Model saving failed**: {str(e)}")
                st.session_state.model_saved_successfully = False
        
        # Render sidebar for predictions
        render_sidebar_prediction(model, feature_columns)
        
        # Main dashboard
        st.header("📊 Model Performance Dashboard")
        
        # Model saving status
        if hasattr(st.session_state, 'model_saved_successfully') and st.session_state.model_saved_successfully:
            with st.expander("💾 Model Saved Successfully", expanded=False):
                st.success("✅ **Model and all related files have been saved successfully!**")
                
                if hasattr(st.session_state, 'saved_model_files'):
                    saved_files = st.session_state.saved_model_files
                    st.markdown("**Saved Files:**")
                    for file_type, file_path in saved_files.items():
                        st.write(f"📄 **{file_type.title()}**: `{file_path}`")
                    
                    st.info("💡 **Tip**: These files contain your trained model, feature names, configuration, and performance metrics. You can use them to load the model in other applications or share with others.")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="ROC-AUC Score",
                value=f"{evaluation_results['roc_auc']:.4f}",
                help="Area Under the ROC Curve - measures model's ability to distinguish between classes"
            )
        
        with col2:
            st.metric(
                label="Cross-Val Mean",
                value=f"{evaluation_results['cv_mean']:.4f}",
                delta=f"±{evaluation_results['cv_std']:.4f}",
                help="Cross-validation performance with standard deviation"
            )
        
        with col3:
            st.metric(
                label="Precision",
                value=f"{evaluation_results['precision']:.4f}",
                help="Precision for default class (positive predictive value)"
            )
        
        with col4:
            st.metric(
                label="Recall",
                value=f"{evaluation_results['recall']:.4f}",
                help="Recall for default class (sensitivity)"
            )
        
        # Visualization row
        st.header("📈 Model Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            cm_fig = create_confusion_matrix_plot(evaluation_results['confusion_matrix'])
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            # ROC Curve
            roc_fig = create_roc_curve_plot(evaluation_results['roc_curve_data'])
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Feature Importance Section
        st.header("🎯 Feature Importance Analysis")
        
        tab1, tab2 = st.tabs(["LightGBM Feature Importance", "SHAP Analysis"])
        
        with tab1:
            # LightGBM built-in feature importance
            st.subheader("Model Feature Importance")
            st.markdown("Features ranked by their importance in the LightGBM model:")
            
            importance_fig = create_feature_importance_plot(evaluation_results['feature_importance'])
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Feature importance table
            with st.expander("📋 View Feature Importance Table"):
                st.dataframe(
                    evaluation_results['feature_importance'].head(30),
                    use_container_width=True
                )
        
        with tab2:
            # SHAP Analysis
            st.subheader("SHAP Feature Importance")
            st.markdown("SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions:")
            
            try:
                with st.spinner("Calculating SHAP values... This may take a moment."):
                    # Calculate SHAP values on a sample for performance
                    shap_values = calculate_shap_values(model, X, max_samples=500)
                    
                    shap_fig = create_shap_summary_plot(shap_values, X)
                    if shap_fig:
                        st.plotly_chart(shap_fig, use_container_width=True)
                    
                    st.info("💡 **SHAP Interpretation**: Higher values indicate features that contribute more to prediction variance. " +
                           "This helps understand which features are most influential in determining default risk.")
                    
            except Exception as e:
                st.warning(f"⚠️ SHAP analysis unavailable: {str(e)}")
                st.info("SHAP analysis requires additional computational resources. The model feature importance above provides similar insights.")
        
        # Data Overview Section
        with st.expander("📋 Dataset Overview"):
            st.subheader("Dataset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Information:**")
                st.write(f"- **Total Records**: {len(merged_df):,}")
                st.write(f"- **Total Features**: {len(feature_columns):,}")
                st.write(f"- **Default Rate**: {y.mean():.2%}")
                st.write(f"- **Missing Values**: {merged_df.isnull().sum().sum():,}")
            
            with col2:
                # Target distribution
                target_counts = y.value_counts()
                fig_target = px.pie(
                    values=target_counts.values,
                    names=['No Default', 'Default'],
                    title="Target Distribution",
                    color_discrete_map={'No Default': '#28a745', 'Default': '#dc3545'}
                )
                st.plotly_chart(fig_target, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Built with ❤️ using Streamlit, LightGBM, and SHAP</p>
            <p>Home Credit Default Risk Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ **Application Error**: {str(e)}")
        st.info("Please check your data files and try again.")


if __name__ == "__main__":
    main()
