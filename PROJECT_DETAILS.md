# Project Details

## Project Overview
This is a **Churn Prediction** system for telecommunications companies using Machine Learning and Streamlit.

## What's Included

### Code Files
1. **ChurnPrediction.py** - Main Streamlit application for interactive predictions and analytics
2. **modeltrain.py** - Model training script using Logistic Regression
3. **Churn Prediction.ipynb** - Jupyter notebook with Exploratory Data Analysis (EDA)
4. **Model_metrics.ipynb** - Jupyter notebook with model evaluation and metrics

### Data
- **WA_Fn-UseC_-Telco-Customer-Churn.csv** - Kaggle Telco Customer Churn dataset (7,043 records)

### Models
- **model_C=1.0.bin** - Pre-trained Logistic Regression model (serialized with pickle)

### Configuration
- **requirements.txt** - Python dependencies
- **.gitignore** - Git ignore rules
- **README.md** - Comprehensive project documentation

## Key Features

✨ **Interactive Dashboard**
- Real-time churn predictions
- Customer analytics and statistics
- Beautiful Streamlit UI

🤖 **Machine Learning**
- Logistic Regression classifier
- 5-fold cross-validation
- ROC-AUC performance metric

📊 **Data Visualization**
- Churn distribution charts
- Feature analysis by customer segments
- Interactive Plotly visualizations

## Technology Stack
- Python 3.8+
- Streamlit (web framework)
- Scikit-learn (ML library)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn, Plotly (visualization)

## Dataset Information
**Source**: Kaggle Telco Customer Churn

**Features** (21):
- Customer demographics (gender, age, partnership, etc.)
- Services (phone, internet, security, etc.)
- Contract details and billing information
- Target: Churn (Yes/No)

**Records**: 7,043 customers
