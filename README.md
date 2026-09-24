# Stroke Prediction System using Logistic Regression

A machine learning web application that predicts stroke risk based on patient clinical data using Logistic Regression, deployed via Streamlit.

---

## Project Overview

This project is a binary classification system that predicts whether a patient is at risk of stroke based on 11 clinical features. The model is trained on the Kaggle Stroke Prediction Dataset and deployed as an interactive web application.

---

## Dataset

- Source: Kaggle — Stroke Prediction Dataset
- Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Patients: 5,110
- Features: 12 (including target)
- Target: stroke (0 = No, 1 = Yes)
- Class Imbalance: 95% No Stroke / 5% Stroke

---

## Features Used

- age
- gender
- hypertension
- heart_disease
- ever_married
- work_type
- Residence_type
- avg_glucose_level
- bmi
- smoking_status
- stroke (target)

---

## Methodology (KDD Process)

1. Data Acquisition — Kaggle stroke dataset
2. Data Preprocessing
   - Missing BMI values filled with median
   - ID column dropped
   - One-hot encoding for categorical columns
   - StandardScaler for feature normalization
   - SMOTE for class imbalance handling
3. Model Training — Logistic Regression (Scikit-learn)
4. Model Evaluation — Confusion Matrix, Accuracy, Precision, Recall, AUC
5. Deployment — Streamlit web application

---

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~78%   |
| Precision | ~XX%   |
| Recall    | ~XX%   |
| AUC       | ~0.85  |

---

## Project Structure
