import streamlit as st
import numpy as np
import pickle

# ==============================
# LOAD MODELS
# ==============================
model_class = pickle.load(open("model_class.pkl", "rb"))
model_reg   = pickle.load(open("model_reg.pkl",   "rb"))
scaler      = pickle.load(open("scaler.pkl",       "rb"))

# ==============================
# PAGE SETTINGS
# ==============================
st.set_page_config(page_title="LungGuard — Risk Detector", layout="centered")
st.title("🫁 Lung Cancer Risk Prediction System")
st.write("Fill in the patient details below to assess risk level.")

# ==============================
# IMPORTANT NOTE FOR USERS
# ==============================
st.info(
    "ℹ️ **Scale:** All symptom/factor sliders use a 1–8 scale "
    "(matching the dataset). 1 = minimal, 8 = severe/maximum."
)

# ==============================
# INPUT FIELDS
# ==============================
st.subheader("Demographics")
col1, col2 = st.columns(2)
Age    = col1.slider("Age", 1, 80, 30)
Gender = col2.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

st.subheader("Environmental & Lifestyle")
col1, col2, col3 = st.columns(3)
Air_Pollution          = col1.slider("Air Pollution",       1, 8, 4)
Alcohol_use            = col2.slider("Alcohol Use",         1, 8, 4)
Smoking                = col3.slider("Smoking",             1, 8, 4)
col1, col2, col3 = st.columns(3)
Passive_Smoker         = col1.slider("Passive Smoker",      1, 8, 4)
Dust_Allergy           = col2.slider("Dust Allergy",        1, 8, 4)
OccuPational_Hazards   = col3.slider("Occupational Hazards",1, 8, 4)
col1, col2 = st.columns(2)
Obesity                = col1.slider("Obesity",             1, 8, 4)
Balanced_Diet          = col2.slider("Balanced Diet",       1, 8, 4)

st.subheader("Medical & Genetic")
col1, col2 = st.columns(2)
Genetic_Risk           = col1.slider("Genetic Risk",        1, 8, 4)
chronic_Lung_Disease   = col2.slider("Chronic Lung Disease",1, 8, 4)

st.subheader("Symptoms")
col1, col2, col3 = st.columns(3)
Chest_Pain             = col1.slider("Chest Pain",          1, 8, 4)
Coughing_of_Blood      = col2.slider("Coughing of Blood",   1, 8, 4)
Fatigue                = col3.slider("Fatigue",             1, 8, 4)
col1, col2, col3 = st.columns(3)
Weight_Loss            = col1.slider("Weight Loss",         1, 8, 4)
Shortness_of_Breath    = col2.slider("Shortness of Breath", 1, 8, 4)
Wheezing               = col3.slider("Wheezing",            1, 8, 4)
col1, col2, col3 = st.columns(3)
Swallowing_Difficulty  = col1.slider("Swallowing Difficulty",1, 8, 4)
Clubbing_of_Finger_Nails = col2.slider("Clubbing of Nails", 1, 8, 4)
Frequent_Cold          = col3.slider("Frequent Cold",       1, 8, 4)
col1, col2 = st.columns(2)
Dry_Cough              = col1.slider("Dry Cough",           1, 8, 4)
Snoring                = col2.slider("Snoring",             1, 8, 4)

# ==============================
# FEATURE ARRAY (ORDER MUST MATCH TRAINING)
# ==============================
features = np.array([[
    Age, Gender, Air_Pollution, Alcohol_use, Dust_Allergy,
    OccuPational_Hazards, Genetic_Risk, chronic_Lung_Disease,
    Balanced_Diet, Obesity, Smoking, Passive_Smoker,
    Chest_Pain, Coughing_of_Blood, Fatigue, Weight_Loss,
    Shortness_of_Breath, Wheezing, Swallowing_Difficulty,
    Clubbing_of_Finger_Nails, Frequent_Cold, Dry_Cough, Snoring
]])

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict Risk"):

    features_scaled = scaler.transform(features)

    # Risk % from Linear Regression
    risk_percentage = model_reg.predict(features_scaled)[0]
    risk_percentage = float(np.clip(risk_percentage, 0, 100))

    # Category derived from score — guaranteed consistent, no contradiction
    if risk_percentage < 33:
        risk_level = "Low"
    elif risk_percentage < 66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # ==============================
    # MEDICAL RECOMMENDATIONS
    # ==============================
    precautions = {
        "Low": [
            "Maintain a healthy balanced diet",
            "Avoid smoking and pollution exposure",
            "Exercise regularly (30 min/day)",
            "Schedule annual health check-ups"
        ],
        "Medium": [
            "Reduce smoking and alcohol consumption",
            "Schedule medical checkups every 6 months",
            "Monitor respiratory symptoms closely",
            "Improve diet and increase physical activity",
            "Consider a pulmonary function test"
        ],
        "High": [
            "Consult a pulmonologist immediately",
            "Undergo CT scan / chest X-ray screening",
            "Stop smoking — critical priority",
            "Follow all medical advice strictly",
            "Inform family; consider genetic counselling",
            "Keep a daily symptom journal"
        ]
    }

    # ==============================
    # OUTPUT
    # ==============================
    color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_level]

    st.markdown("---")
    st.subheader("Results")
    st.markdown(f"### Risk Level: :{color}[{risk_level}]")
    st.metric("Risk Score", f"{risk_percentage:.1f} / 100")
    st.progress(int(risk_percentage) / 100)

    st.write("### Recommendations:")
    for tip in precautions[risk_level]:
        st.write(f"- {tip}")

    st.warning("⚠️ This is not a medical diagnosis. Please consult a healthcare professional.")
