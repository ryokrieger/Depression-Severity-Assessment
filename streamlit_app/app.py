import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# CONFIG
st.set_page_config(page_title="Depression Prediction System", layout="centered")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ml2" / "mir" / "logistic_regression.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

# LOAD MODEL
model = joblib.load(MODEL_PATH)

# FEATURE ORDER
FEATURE_COLS = [
    "PSS4", "PSS10", "PSS2", "PSS1", "PSS9", "PSS3", "PSS5",
    "PHQ2", "PHQ6", "PHQ4", "PHQ7", "PHQ9", "PHQ5"
]

# LABEL MAPPING
LABEL_MAP = {
    0: "No Depression",
    1: "Mild Depression",
    2: "Moderate Depression",
    3: "Moderately Severe Depression",
    4: "Severe Depression"
}

# STREAMLIT UI
st.title("🎓 AI-Based Depression Assessment System")
st.markdown("This system predicts **Depression Level** based on validated **PSS & PHQ questionnaires**.")

st.header("📊 Stress (PSS) Questionnaire")

PSS1 = st.slider("PSS1: Upset due to academic affairs", 0, 4, 0)
PSS2 = st.slider("PSS2: Unable to control important things", 0, 4, 0)
PSS3 = st.slider("PSS3: Nervous & stressed due to academics", 0, 4, 0)
PSS4 = st.slider("PSS4: Unable to cope with workload", 0, 4, 0)
PSS5 = st.slider("PSS5: Confident handling problems", 0, 4, 0)
PSS9 = st.slider("PSS9: Angered due to poor performance", 0, 4, 0)
PSS10 = st.slider("PSS10: Academic difficulties piled up", 0, 4, 0)

st.header("🧠 Depression (PHQ) Questionnaire")

PHQ2 = st.slider("PHQ2: Feeling down or hopeless", 0, 4, 0)
PHQ4 = st.slider("PHQ4: Feeling tired or low energy", 0, 4, 0)
PHQ5 = st.slider("PHQ5: Poor appetite or overeating", 0, 4, 0)
PHQ6 = st.slider("PHQ6: Feeling bad about yourself", 0, 4, 0)
PHQ7 = st.slider("PHQ7: Trouble concentrating", 0, 4, 0)
PHQ9 = st.slider("PHQ9: Thoughts of self-harm", 0, 4, 0)

# PREDICTION
if st.button("🔍 Predict Depression Level"):

    input_data = pd.DataFrame([[
        PSS4, PSS10, PSS2, PSS1, PSS9, PSS3, PSS5,
        PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
    ]], columns=FEATURE_COLS)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Depression Level: **{LABEL_MAP[prediction]}**")

    prob_df = pd.DataFrame({
        "Depression Level": list(LABEL_MAP.values()),
        "Probability": probabilities
    })

    st.subheader("📈 Prediction Probability Distribution")
    st.bar_chart(prob_df.set_index("Depression Level"))