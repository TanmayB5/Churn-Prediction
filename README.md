# Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)

**Predict customer churn in telecommunications using Machine Learning**

[Demo](#demo-) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Dataset](#dataset) • [Model](#model)

</div>

---

## Introduction

In an increasingly connected world, telecommunications companies play a vital role in people's lives. However, competition is fierce and retaining customers has become a difficult task. **Churn** (customer abandonment) is a common problem in the industry that can be costly in terms of lost revenue and reputation. 

This project uses **machine learning techniques** to predict which customers are at risk of leaving, enabling companies to take proactive measures to improve customer retention.

---

## Features

✨ **Interactive Streamlit Dashboard**
- Real-time customer churn predictions
- Input customer data and get instant predictions
- Beautiful, responsive user interface with dark theme

📊 **Comprehensive Analytics**
- Customer retention vs churn metrics
- Statistical distributions by customer features
- Visual analytics with bar charts and pie charts
- Insights on gender, partnership, internet service, contracts, etc.

🤖 **Machine Learning Model**
- Logistic Regression classifier
- Cross-validated with KFold (5 splits)
- Pre-trained model included
- Optimized C parameter (C=1.0)

📈 **Jupyter Notebooks**
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Model metrics and performance analysis

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/TanmayB5/Churn-Prediction.git
cd Churn-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
streamlit --version
```

---

## Usage

### Running the Streamlit App

```bash
streamlit run ChurnPrediction.py
```

The app will start and be accessible at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://localhost:8501 (network access)

### Making Predictions

1. Open the Streamlit app in your browser
2. Navigate to the prediction section
3. Enter customer information:
   - Demographics (gender, age, partnership status)
   - Services (phone, internet, online security, etc.)
   - Contract and billing information
   - Monetary information (monthly charges, total charges)
4. Click "Predict" to see the churn probability
5. Explore the analytics dashboard for statistical insights

### Training the Model

To retrain the model with new data:

```bash
python modeltrain.py
```

This script:
- Loads the Telco customer churn dataset
- Preprocesses and cleans the data
- Trains a Logistic Regression model
- Performs cross-validation (5-fold)
- Saves the trained model as `model_C=1.0.bin`

---

## Dataset

**Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

**Filename**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Features** (21 columns):
- **Customer Info**: customerID, gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Info**: Contract, PaperlessBilling, PaymentMethod, Tenure, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

**Records**: 7,043 customers

---

## Project Structure

```
Churn_Prediction_Streamlit-main/
├── Churn Prediction.ipynb          # Exploratory Data Analysis notebook
├── Model_metrics.ipynb             # Model evaluation and metrics
├── ChurnPrediction.py              # Streamlit application
├── modeltrain.py                   # Model training script
├── model_C=1.0.bin                 # Pre-trained model (pickle format)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Model

### Algorithm
**Logistic Regression** - Binary classification model

### Training Details
- **Training Set Size**: 80% (5,634 customers)
- **Test Set Size**: 20% (1,409 customers)
- **Cross-Validation**: 5-fold KFold
- **Regularization Parameter (C)**: 1.0
- **Evaluation Metric**: ROC-AUC Score

### Performance
The model evaluates customer churn risk with competitive performance. For detailed metrics, see `Model_metrics.ipynb`.

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Programming language |
| **Streamlit** | Interactive web application framework |
| **Scikit-learn** | Machine Learning & model evaluation |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **Plotly** | Interactive visualization |

---

## File Descriptions

### ChurnPrediction.py
Main Streamlit application that provides:
- Interactive user interface for predictions
- Dashboard with customer statistics
- Visualizations of churn distribution
- Real-time prediction interface

### modeltrain.py
Model training pipeline:
- Data loading and preprocessing
- Feature engineering
- Logistic Regression training
- Model serialization with pickle

### Jupyter Notebooks
- **Churn Prediction.ipynb**: Data exploration and analysis
- **Model_metrics.ipynb**: Model evaluation and performance metrics

---

## Getting Started

### 1. Quick Start (3 steps)
```bash
git clone https://github.com/TanmayB5/Churn-Prediction.git
cd Churn-Prediction
pip install -r requirements.txt
streamlit run ChurnPrediction.py
```

### 2. Make Your First Prediction
- Open http://localhost:8501
- Fill in customer details in the sidebar
- Click predict to see churn risk

### 3. Explore the Analytics
- View overall retention metrics
- Analyze churn patterns by customer segment
- Understand feature importance from visualizations

---

## Model Workflow

```
Raw Data
   ↓
Data Cleaning & Preprocessing
   ├─ Handle missing values
   ├─ Normalize numerical features
   └─ Encode categorical variables
   ↓
Feature Engineering
   ├─ DictVectorizer for categorical conversion
   └─ Standardization
   ↓
Model Training (Logistic Regression)
   ├─ 5-Fold Cross-Validation
   └─ Hyperparameter Tuning (C=1.0)
   ↓
Model Evaluation
   ├─ ROC-AUC Score
   ├─ Classification Metrics
   └─ Performance Analysis
   ↓
Deployment (Streamlit App)
   └─ Interactive Predictions & Analytics
```

---

## Requirements

See `requirements.txt`:
- streamlit
- pandas
- numpy
- seaborn
- scikit-learn
- matplotlib
- plotly

---

## Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

---


## Author

**Tanmay B**

- GitHub: [@TanmayB5](https://github.com/TanmayB5)
- Project Link: [Churn Prediction](https://github.com/TanmayB5/Churn-Prediction)

---

## Acknowledgments

- Dataset source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Special thanks to the open-source community for the amazing libraries used in this project
- Inspired by real-world customer retention challenges in telecommunications

---

## Support

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🔗 Sharing with others
- 📧 Providing feedback
- 🤝 Contributing to improvements

For questions or issues, please open an [issue](https://github.com/TanmayB5/Churn-Prediction/issues) on GitHub.

---

**Last Updated**: February 2026
