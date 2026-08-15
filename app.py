import streamlit as st
import pandas as pd
import joblib
import numpy as np

# page title and icon
st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide"
)

# loading model and scaler i saved from the notebook
model = joblib.load(r'C:\Users\Ar\Desktop\stroke_prediction\model.pkl')
scaler = joblib.load(r'C:\Users\Ar\Desktop\stroke_prediction\scaler.pkl')

st.title("🧠 Stroke Risk Prediction System")
st.write("Enter patient details to predict stroke risk using Linear Regression")

st.markdown("---")

# dividing screen into two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    # sliders are easier for doctors to use than typing numbers
    age = st.slider("Age", 28, 77, 50)
    resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.slider("Cholesterol", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 202, 150)
    oldpeak = st.slider("Oldpeak", -2.6, 6.2, 0.0, 0.1)
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    st.subheader("Medical History")
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# everything below runs only when the button is clicked
if st.button("🔍 Predict Stroke Risk", use_container_width=True):

    # converting text to 0 or 1 because the model was trained on numbers
    sex_m = 1 if sex == "Male" else 0
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    exercise_angina_y = 1 if exercise_angina == "Yes" else 0

    # chest pain has 4 types, ASY was dropped during training so no column for it
    chest_pain_ata = 1 if chest_pain == "ATA" else 0
    chest_pain_nap = 1 if chest_pain == "NAP" else 0
    chest_pain_ta = 1 if chest_pain == "TA" else 0

    # LVH was dropped during training so no column for it
    resting_ecg_normal = 1 if resting_ecg == "Normal" else 0
    resting_ecg_st = 1 if resting_ecg == "ST" else 0

    # Down was dropped during training so no column for it
    st_slope_flat = 1 if st_slope == "Flat" else 0
    st_slope_up = 1 if st_slope == "Up" else 0

    # column order here must match exactly what was used in training
    # if order is wrong the model gives wrong predictions silently
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

    # using the same scaler from training so values are scaled the same way
    input_scaled = scaler.transform(input_data)

    # linear regression gives a raw number not 0 or 1
    # i clip it between 0 and 1 and use 0.5 as the cutoff
    raw = model.predict(input_scaled)[0]
    prediction = 1 if raw >= 0.5 else 0
    probability = float(np.clip(raw, 0, 1))

    st.markdown("---")
    st.subheader("Prediction Result")

    risk_percentage = round(probability * 100, 2)

    # red for high risk, green for low risk
    if prediction == 1:
        st.error("⚠️ High Stroke Risk Detected — " + str(risk_percentage) + "% probability")
        st.write("This patient shows high risk factors. Please consult a medical professional immediately.")
    else:
        st.success("✅ Low Stroke Risk — " + str(risk_percentage) + "% probability")
        st.write("This patient shows low risk based on entered details. Regular checkups are still recommended.")

    # visual bar to show the risk level
    st.progress(int(risk_percentage))

    st.markdown("---")
    st.subheader("Patient Summary")

    # showing 4 key values as cards
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.metric("Age", str(age) + " years")
    with col4:
        st.metric("Cholesterol", str(cholesterol) + " mg/dL")
    with col5:
        st.metric("Resting BP", str(resting_bp) + " mmHg")
    with col6:
        st.metric("Max Heart Rate", str(max_hr) + " bpm")

    st.markdown("---")
    # always important to add disclaimer in healthcare apps
    st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Always consult a qualified healthcare provider.")