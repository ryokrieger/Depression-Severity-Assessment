import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import os

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ml2" / "mir"/ "logistic_regression.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "features2" / "mir" / "scaler.pkl"

# LOAD MODEL & SCALER
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CLASS LABELS
class_names = {
    0: "Minimal",
    1: "Mild",
    2: "Moderate",
    3: "Moderately Severe",
    4: "Severe"
}

# STREAMLIT UI
st.set_page_config(page_title="Depression Prediction (AI)", layout="centered")

st.title("🧠 AI-Based Depression Severity Prediction")
st.markdown("This system predicts **depression severity** using validated stress and depression questionnaire responses.")

st.markdown("---")

# STRESS (PSS) INPUTS
st.subheader("🟡 Perceived Stress Scale (PSS)")

PSS4 = st.slider("PSS4: Unable to cope with mandatory activities", 0, 4, 0)
PSS10 = st.slider("PSS10: Difficulties piling up too high", 0, 4, 0)
PSS2 = st.slider("PSS2: Unable to control important things", 0, 4, 0)
PSS1 = st.slider("PSS1: Upset due to unexpected events", 0, 4, 0)
PSS9 = st.slider("PSS9: Angered by uncontrollable setbacks", 0, 4, 0)
PSS3 = st.slider("PSS3: Feeling nervous and stressed", 0, 4, 0)
PSS5 = st.slider("PSS5: Confident in handling problems", 0, 4, 0)

# DEPRESSION (PHQ) INPUTS
st.subheader("🔵 Patient Health Questionnaire (PHQ)")

PHQ2 = st.slider("PHQ2: Feeling down or depressed", 0, 3, 0)
PHQ6 = st.slider("PHQ6: Feeling like a failure", 0, 3, 0)
PHQ4 = st.slider("PHQ4: Feeling tired or low energy", 0, 3, 0)
PHQ7 = st.slider("PHQ7: Trouble concentrating", 0, 3, 0)
PHQ9 = st.slider("PHQ9: Thoughts of self-harm", 0, 3, 0)
PHQ5 = st.slider("PHQ5: Poor appetite or overeating", 0, 3, 0)

# FEATURE VECTOR
input_features = np.array([[
    PSS4, PSS10, PSS2, PSS1, PSS9, PSS3, PSS5,
    PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
]])

# PREDICTION
if st.button("🔍 Predict Depression Severity"):
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    predicted_label = class_names[prediction]

    st.markdown("---")
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Depression Level: **{predicted_label}**")


    # PROBABILITY DISPLAY
    st.subheader("📊 Prediction Probability Distribution")

    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob:.2%}")

    st.bar_chart(probabilities)

# FOOTER
st.markdown("---")
st.caption("⚠️ This tool is for research and educational purposes only. It is not a medical diagnosis system.")