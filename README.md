# Stroke Prediction System using Logistic Regression

A machine learning web application that predicts stroke risk based on patient clinical data, deployed via Streamlit.

## Dataset
- Kaggle Stroke Prediction Dataset
- 5,110 patients, 12 features
- Target: stroke (0 = No, 1 = Yes)

## Methodology
- Missing BMI values filled with median
- One-hot encoding for categorical columns
- SMOTE for class imbalance
- StandardScaler for normalization
- Logistic Regression (Scikit-learn)

## Project Structure
