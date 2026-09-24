import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: #f0f4f8; }

    [data-testid="stSidebar"] { display: none; }

    #MainMenu, footer, header { visibility: hidden; }

    /* NAV */
    .navbar {
        background: white;
        padding: 16px 40px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-brand {
        font-size: 20px;
        font-weight: 700;
        color: #e05c1a;
    }
    .nav-links {
        font-size: 14px;
        color: #666;
    }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #e05c1a 0%, #c94400 100%);
        border-radius: 20px;
        padding: 50px 40px;
        color: white;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        right: -40px;
        top: -40px;
        width: 250px;
        height: 250px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        right: 60px;
        bottom: -60px;
        width: 180px;
        height: 180px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-badge {
        background: rgba(255,255,255,0.2);
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 12px;
        display: inline-block;
        margin-bottom: 16px;
        letter-spacing: 1px;
    }
    .hero-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    .hero-desc {
        font-size: 15px;
        opacity: 0.88;
        max-width: 520px;
        line-height: 1.7;
        margin-bottom: 28px;
    }
    .hero-stats {
        display: flex;
        gap: 32px;
    }
    .hero-stat-num {
        font-size: 26px;
        font-weight: 700;
    }
    .hero-stat-label {
        font-size: 12px;
        opacity: 0.8;
    }

    /* CARDS */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        border-left: 4px solid #e05c1a;
    }
    .info-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
    }
    .info-card-text {
        font-size: 14px;
        color: #666;
        line-height: 1.6;
    }

    /* FORM SECTION */
    .form-section {
        background: white;
        border-radius: 20px;
        padding: 36px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        margin-bottom: 24px;
    }
    .form-section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 6px;
    }
    .form-section-sub {
        font-size: 13px;
        color: #888;
        margin-bottom: 28px;
    }
    .form-label {
        font-size: 13px;
        font-weight: 600;
        color: #444;
        margin-bottom: 4px;
    }
    .section-divider {
        font-size: 13px;
        font-weight: 600;
        color: #e05c1a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #f0e0d8;
    }

    /* RISK FACTORS */
    .risk-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .risk-high {
        border-left: 4px solid #e05c1a;
    }
    .risk-low {
        border-left: 4px solid #22c55e;
    }
    .risk-tag {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 50px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .tag-high { background: #fff0eb; color: #e05c1a; }
    .tag-low { background: #f0fdf4; color: #16a34a; }

    /* RESULT */
    .result-high {
        background: linear-gradient(135deg, #e05c1a, #c94400);
        border-radius: 20px;
        padding: 36px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .result-low {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 20px;
        padding: 36px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .result-emoji { font-size: 52px; margin-bottom: 12px; }
    .result-title { font-size: 26px; font-weight: 700; margin-bottom: 8px; }
    .result-pct { font-size: 52px; font-weight: 800; margin: 8px 0; }
    .result-sub { font-size: 14px; opacity: 0.9; line-height: 1.6; }

    /* METRIC CARDS */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 20px;
    }
    .metric-box {
        background: white;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #e05c1a;
    }
    .metric-lbl {
        font-size: 11px;
        color: #888;
        margin-top: 4px;
    }

    /* PROGRESS BAR */
    .progress-wrap {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .progress-label {
        font-size: 13px;
        color: #666;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .progress-bar-bg {
        background: #f0f0f0;
        border-radius: 50px;
        height: 12px;
    }
    .progress-bar-fill-high {
        background: linear-gradient(90deg, #e05c1a, #c94400);
        height: 12px;
        border-radius: 50px;
        transition: width 0.5s;
    }
    .progress-bar-fill-low {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        height: 12px;
        border-radius: 50px;
        transition: width 0.5s;
    }

    /* DISCLAIMER */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 12px;
        color: #92400e;
        margin-top: 20px;
    }

    /* STROKE INFO */
    .stroke-info {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* PREDICT BUTTON */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #e05c1a, #c94400) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 40px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin-top: 10px !important;
        box-shadow: 0 4px 16px rgba(224,92,26,0.3) !important;
    }
    div[data-testid="stButton"] > button p {
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #c94400, #a83800) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(224,92,26,0.4) !important;
    }

    /* STREAMLIT OVERRIDES */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border-color: #e8e8e8 !important;
    }
    .stSlider > div > div > div {
        background: #e05c1a !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }
    .stRadio > div {
        gap: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_model():
    model = joblib.load(r'C:\Users\Ar\Desktop\stroke_prediction\model.pkl')
    scaler = joblib.load(r'C:\Users\Ar\Desktop\stroke_prediction\scaler.pkl')
    return model, scaler

model, scaler = load_model()

# NAVBAR
st.markdown("""
<div class='navbar'>
    <div class='nav-brand'>🧠 StrokeGuard</div>
    <div class='nav-links'>Stroke Risk Prediction System &nbsp;·&nbsp; Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🏥 CLINICAL DECISION SUPPORT</div>
    <div class='hero-title'>Stroke Risk<br>Prediction System</div>
    <div class='hero-desc'>
        An AI-powered tool using Logistic Regression to assess stroke risk 
        from patient health data. Enter patient details below to get 
        an instant, evidence-based risk assessment.
    </div>
    <div class='hero-stats'>
        <div>
            <div class='hero-stat-num'>5,110</div>
            <div class='hero-stat-label'>Patients Trained On</div>
        </div>
        <div>
            <div class='hero-stat-num'>12</div>
            <div class='hero-stat-label'>Clinical Features</div>
        </div>
        <div>
            <div class='hero-stat-num'>~78%</div>
            <div class='hero-stat-label'>Model Accuracy</div>
        </div>
        <div>
            <div class='hero-stat-num'>LR</div>
            <div class='hero-stat-label'>Algorithm</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["🔍 Predict Risk", "ℹ️ About Stroke", "⚠️ Risk Factors"])

# ═══════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════
with tab1:
    st.markdown("<div class='form-section'>", unsafe_allow_html=True)
    st.markdown("<div class='form-section-title'>Patient Risk Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='form-section-sub'>Fill in all patient details accurately for the best prediction results.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-divider'>Personal Info</div>", unsafe_allow_html=True)
        age = st.slider("Age", 1, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.selectbox("Marital Status", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

    with col2:
        st.markdown("<div class='section-divider'>Medical History</div>", unsafe_allow_html=True)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        avg_glucose_level = st.slider("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, 0.1)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)

    with col3:
        st.markdown("<div class='section-divider'>Lifestyle</div>", unsafe_allow_html=True)
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#fff8f5; border-radius:12px; padding:16px;'>
            <div style='font-size:12px; color:#888; margin-bottom:8px;'>Quick Reference</div>
            <div style='font-size:13px; color:#444; line-height:1.8;'>
                🔴 BMI > 30 → Obese<br>
                🟡 Glucose > 140 → High<br>
                🔴 Age > 60 → Higher risk<br>
                🔴 Smoking → Major risk
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Stroke Risk", use_container_width=True)

    if predict_btn:
        # encoding
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0
        gender_male = 1 if gender == "Male" else 0
        gender_other = 1 if gender == "Other" else 0
        ever_married_yes = 1 if ever_married == "Yes" else 0
        work_never = 1 if work_type == "Never_worked" else 0
        work_private = 1 if work_type == "Private" else 0
        work_self = 1 if work_type == "Self-employed" else 0
        work_children = 1 if work_type == "children" else 0
        residence_urban = 1 if residence_type == "Urban" else 0
        smoking_formerly = 1 if smoking_status == "formerly smoked" else 0
        smoking_never = 1 if smoking_status == "never smoked" else 0
        smoking_smokes = 1 if smoking_status == "smokes" else 0

        input_data = pd.DataFrame([[
            age, hypertension_val, heart_disease_val,
            avg_glucose_level, bmi,
            gender_male, gender_other,
            ever_married_yes,
            work_never, work_private, work_self, work_children,
            residence_urban,
            smoking_formerly, smoking_never, smoking_smokes
        ]], columns=[
            'age', 'hypertension', 'heart_disease',
            'avg_glucose_level', 'bmi',
            'gender_Male', 'gender_Other',
            'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private',
            'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban',
            'smoking_status_formerly smoked',
            'smoking_status_never smoked',
            'smoking_status_smokes'
        ])

        input_scaled = scaler.transform(input_data)
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        risk_pct = round(probability * 100, 2)

        st.markdown("---")

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            if prediction == 1:
                st.markdown(f"""
                <div class='result-high'>
                    <div class='result-emoji'>⚠️</div>
                    <div class='result-title'>High Stroke Risk</div>
                    <div class='result-pct'>{risk_pct}%</div>
                    <div class='result-sub'>
                        This patient shows significant stroke risk.<br>
                        Immediate medical consultation recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                    <div class='result-emoji'>✅</div>
                    <div class='result-title'>Low Stroke Risk</div>
                    <div class='result-pct'>{risk_pct}%</div>
                    <div class='result-sub'>
                        This patient shows low stroke risk.<br>
                        Regular checkups are still recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            bar_color = "high" if prediction == 1 else "low"
            st.markdown(f"""
            <div class='progress-wrap'>
                <div class='progress-label'>
                    <span>Risk Level</span>
                    <span style='font-weight:600; color:{"#e05c1a" if prediction==1 else "#16a34a"}'>{risk_pct}%</span>
                </div>
                <div class='progress-bar-bg'>
                    <div class='progress-bar-fill-{bar_color}' style='width:{risk_pct}%;'></div>
                </div>
                <div style='display:flex; justify-content:space-between; font-size:11px; color:#aaa; margin-top:8px;'>
                    <span>Low Risk</span>
                    <span>Moderate</span>
                    <span>High Risk</span>
                </div>
            </div>

            <div class='progress-wrap'>
                <div style='font-size:13px; font-weight:600; color:#444; margin-bottom:14px;'>Patient Summary</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                    <div style='background:#f8f8f8; border-radius:10px; padding:12px; text-align:center;'>
                        <div style='font-size:20px; font-weight:700; color:#e05c1a;'>{age}</div>
                        <div style='font-size:11px; color:#888;'>Age</div>
                    </div>
                    <div style='background:#f8f8f8; border-radius:10px; padding:12px; text-align:center;'>
                        <div style='font-size:20px; font-weight:700; color:#e05c1a;'>{bmi}</div>
                        <div style='font-size:11px; color:#888;'>BMI</div>
                    </div>
                    <div style='background:#f8f8f8; border-radius:10px; padding:12px; text-align:center;'>
                        <div style='font-size:20px; font-weight:700; color:#e05c1a;'>{avg_glucose_level}</div>
                        <div style='font-size:11px; color:#888;'>Glucose</div>
                    </div>
                    <div style='background:#f8f8f8; border-radius:10px; padding:12px; text-align:center;'>
                        <div style='font-size:20px; font-weight:700; color:#e05c1a;'>{hypertension}</div>
                        <div style='font-size:11px; color:#888;'>Hypertension</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
            ⚠️ <b>Medical Disclaimer:</b> This tool is for educational purposes only and does not replace 
            professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════
# TAB 2 — ABOUT STROKE
# ═══════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🧠 What is a Stroke?</div>
            <div style='font-size:14px; color:#444; line-height:1.8;'>
                A stroke occurs when blood flow to part of the brain is blocked or a blood vessel bursts.
                Brain cells begin to die within minutes without oxygen. It is a medical emergency
                requiring immediate attention.
            </div>
        </div>

        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>⚡ Warning Signs (FAST)</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                <b style='color:#e05c1a;'>F</b> — Face drooping on one side<br>
                <b style='color:#e05c1a;'>A</b> — Arm weakness or numbness<br>
                <b style='color:#e05c1a;'>S</b> — Speech difficulty or slurred<br>
                <b style='color:#e05c1a;'>T</b> — Time to call emergency services
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>📊 Global Statistics</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🌍 13 million strokes occur globally each year<br>
                💔 5.5 million deaths annually from stroke<br>
                ♿ Leading cause of long-term disability<br>
                ⏰ Every 40 seconds someone has a stroke in the US<br>
                🎯 80% of strokes are preventable
            </div>
        </div>

        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🔬 Types of Stroke</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                <b>Ischemic (87%)</b> — Blood clot blocks artery<br>
                <b>Hemorrhagic (13%)</b> — Blood vessel ruptures<br>
                <b>TIA</b> — Temporary blockage (mini-stroke)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stroke-info'>
        <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🤖 About This Model</div>
        <div style='font-size:14px; color:#444; line-height:1.8;'>
            This system uses <b>Logistic Regression</b> — a well-established machine learning algorithm
            chosen for its high interpretability in healthcare settings. The model was trained on
            5,110 real patient records from Kaggle, with SMOTE applied to handle class imbalance.
            Preprocessing included BMI imputation, one-hot encoding, and StandardScaler normalization.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════
# TAB 3 — RISK FACTORS
# ═══════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='font-size:16px; font-weight:700; color:#e05c1a; margin-bottom:14px;'>⚠️ High Risk Factors</div>
        """, unsafe_allow_html=True)

        factors_high = [
            ("Hypertension", "High blood pressure damages arteries increasing stroke risk significantly."),
            ("High Glucose Level", "Blood sugar above 140 mg/dL increases stroke risk — diabetes is a major factor."),
            ("Old Age (60+)", "Risk doubles every 10 years after age 55."),
            ("Heart Disease", "Irregular heartbeat and other heart conditions increase clot formation."),
            ("Smoking", "Smoking doubles stroke risk by damaging blood vessels and increasing clot formation."),
            ("High BMI", "Obesity (BMI > 30) leads to hypertension and diabetes — both stroke risk factors."),
        ]

        for title, desc in factors_high:
            st.markdown(f"""
            <div class='risk-card risk-high'>
                <span class='risk-tag tag-high'>HIGH RISK</span>
                <div style='font-size:14px; font-weight:600; color:#1a1a1a;'>{title}</div>
                <div style='font-size:13px; color:#666; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='font-size:16px; font-weight:700; color:#16a34a; margin-bottom:14px;'>✅ Protective Factors</div>
        """, unsafe_allow_html=True)

        factors_low = [
            ("Normal Blood Pressure", "Keeping BP below 120/80 mmHg significantly reduces stroke risk."),
            ("Healthy Glucose Level", "Maintaining blood sugar below 100 mg/dL protects brain vessels."),
            ("Young Age", "People under 40 have significantly lower stroke risk."),
            ("Never Smoked", "Non-smokers have half the stroke risk of smokers."),
            ("Healthy BMI (18.5-24.9)", "Maintaining healthy weight reduces hypertension and diabetes risk."),
            ("Active Lifestyle", "Regular exercise reduces stroke risk by up to 27%."),
        ]

        for title, desc in factors_low:
            st.markdown(f"""
            <div class='risk-card risk-low'>
                <span class='risk-tag tag-low'>PROTECTIVE</span>
                <div style='font-size:14px; font-weight:600; color:#1a1a1a;'>{title}</div>
                <div style='font-size:13px; color:#666; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)