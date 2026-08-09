

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide"
)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Stroke Risk Prediction System")
st.write("Enter patient details below to predict stroke risk based on clinical risk factors using Logistic Regression")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    age = st.slider("Age", 28, 77, 50)
    resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.slider("Cholesterol", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 202, 150)
    oldpeak = st.slider("Oldpeak (ST depression)", -2.6, 6.2, 0.0, 0.1)
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    st.subheader("Medical History")
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ============================================
# PART 3: PREDICTION LOGIC
# ============================================

st.markdown("---")

# Predict button — when clicked, runs the model
if st.button("🔍 Predict Stroke Risk", use_container_width=True):

    # ----------------------------------------
    # Convert user inputs into model format
    # ----------------------------------------
    
    # Convert Yes/No and categorical text inputs into 
    # the same encoded format the model was trained on
    sex_m = 1 if sex == "Male" else 0
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    exercise_angina_y = 1 if exercise_angina == "Yes" else 0
    
    # ChestPainType one-hot encoding (ASY is the reference/dropped category)
    chest_pain_ata = 1 if chest_pain == "ATA" else 0
    chest_pain_nap = 1 if chest_pain == "NAP" else 0
    chest_pain_ta = 1 if chest_pain == "TA" else 0
    
    # RestingECG one-hot encoding (LVH is the reference/dropped category)
    resting_ecg_normal = 1 if resting_ecg == "Normal" else 0
    resting_ecg_st = 1 if resting_ecg == "ST" else 0
    
    # ST_Slope one-hot encoding (Down is the reference/dropped category)
    st_slope_flat = 1 if st_slope == "Flat" else 0
    st_slope_up = 1 if st_slope == "Up" else 0
    
    # ----------------------------------------
    # Build input row in EXACT same column order as training data
    # ----------------------------------------
    
    input_data = pd.DataFrame([[
        age, resting_bp, cholesterol, fasting_bs_val, max_hr, oldpeak,
        sex_m, chest_pain_ata, chest_pain_nap, chest_pain_ta,
        resting_ecg_normal, resting_ecg_st, exercise_angina_y,
        st_slope_flat, st_slope_up
    ]], columns=[
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ])
    
    # ----------------------------------------
    # Scale input using the SAME scaler from training
    # ----------------------------------------
    input_scaled = scaler.transform(input_data)
    
    # ----------------------------------------
    # Make prediction
    # ----------------------------------------
    # Linear Regression gives continuous value
# we apply 0.5 threshold to convert to 0 or 1
    raw = model.predict(input_scaled)[0]
    prediction = 1 if raw >= 0.5 else 0
    # Linear Regression has no predict_proba
# so we use raw prediction value as probability
    probability = model.predict(input_scaled)[0]

    # Clip between 0 and 1 because Linear Regression
    # can predict values outside this range
    probability = float(max(0, min(1, probability)))

    # ----------------------------------------
    # PART 4: DISPLAY RESULTS
    # ----------------------------------------
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Convert probability to percentage for display
    risk_percentage = round(probability * 100, 2)
    
    # Show result based on prediction
    if prediction == 1:
        st.error(f"⚠️ High Stroke Risk Detected — {risk_percentage}% probability")
        st.write("This patient shows risk factors associated with cardiovascular disease. Please consult a medical professional for further evaluation.")
    else:
        st.success(f"✅ Low Stroke Risk — {risk_percentage}% probability")
        st.write("This patient shows low risk based on the entered factors. Regular checkups are still recommended.")
    
    # Progress bar to visually show risk percentage
    st.progress(int(risk_percentage))
    
    # ----------------------------------------
    # Show key risk factors entered
    # ----------------------------------------
    
    st.markdown("---")
    st.subheader("Key Risk Factors Summary")
    
    col3, col4, col5, col4_extra = st.columns(4)
    
    with col3:
        st.metric("Age", f"{age} years")
    with col4:
        st.metric("Cholesterol", f"{cholesterol} mg/dL")
    with col5:
        st.metric("Resting BP", f"{resting_bp} mmHg")
    with col4_extra:
        st.metric("Max Heart Rate", f"{max_hr} bpm")
    
    # ----------------------------------------
    # Disclaimer — important for healthcare apps
    # ----------------------------------------
    st.markdown("---")
    st.caption("⚠️ Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.")

    