# preeclamsia-prediction

# Project Overview

This project implements a machine learning–based Maternal Health Risk Prediction System that classifies pregnant women into Low Risk, Mid Risk, or High Risk categories using clinical parameters. The goal is to support early risk assessment and improve maternal healthcare decision-making through data-driven insights.

The system includes data preprocessing, model training, hyperparameter tuning, model explainability, and a user-friendly Gradio web interface for real-time predictions.

# Objectives

Predict maternal health risk levels using clinical data

Compare multiple machine learning models

Improve accuracy using hyperparameter tuning and ensemble learning

Interpret predictions using explainable AI techniques

Provide an interactive interface for real-time predictions

# Technologies & Libraries

Python

Pandas, NumPy – Data processing

Scikit-learn – Model training, evaluation, tuning

XGBoost – Gradient boosting model

Matplotlib – Data visualization

SHAP – Model explainability

Gradio – Interactive web interface

Google Colab – Development environment

# Dataset

Dataset Name: Maternal Health Risk Data Set

Target Variable: RiskLevel (Low Risk, Mid Risk, High Risk)

Features Used:

Age

Systolic Blood Pressure

Diastolic Blood Pressure

Blood Sugar

Body Temperature

Heart Rate

# Methodology

# Data Loading & Exploration

Dataset inspection and summary statistics

Class distribution analysis

# Data Preprocessing

Label encoding of categorical target variable

Train-test split

# Model Training

Random Forest Classifier

XGBoost Classifier

# Model Evaluation

Accuracy score

Classification report

Cross-validation

# Hyperparameter Tuning

RandomizedSearchCV for Random Forest

# Ensemble Learning

Soft Voting Classifier combining Random Forest and XGBoost

# Model Explainability

Feature importance using Random Forest

SHAP values for global and local interpretation

# Deployment

Gradio-based web interface for real-time predictions

# Results

Random Forest Accuracy: High and stable across folds

XGBoost Accuracy: Competitive performance

Ensemble Model: Improved cross-validation accuracy

Key Influential Features: Blood Pressure, Blood Sugar, Age, Heart Rate

# How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/MonicaAishwarya/maternal-health-risk-prediction.git

2️⃣ Install Dependencies
pip install pandas numpy scikit-learn xgboost shap gradio matplotlib

3️⃣ Run the Application
python app.py


(or run the notebook in Google Colab)

# Web Interface

The Gradio interface allows users to input clinical parameters and receive instant maternal health risk predictions:

Age

Blood Pressure

Blood Sugar

Body Temperature

Heart Rate

# Future Enhancements

Integrate deep learning models

Deploy as a full web application

Add database support for patient records

Improve UI and accessibility

Model monitoring and retraining

# Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical advice or diagnosis.
