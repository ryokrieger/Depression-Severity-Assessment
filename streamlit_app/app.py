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
st.markdown("Scale: 0 - Never, 1 - Almost Never, 2 - Sometimes, 3 - Fairly Often, 4 - Very Often")

PSS1 = st.slider("In a semester, how often have you felt upset due to something that happened in your academic affairs?", 0, 4, 0)
PSS2 = st.slider("In a semester, how often did you feel as if you were unable to control important things in your academic affairs?", 0, 4, 0)
PSS3 = st.slider("In a semester, how often did you feel nervous and stressed because of academic pressure?", 0, 4, 0)
PSS4 = st.slider("In a semester, how often did you feel as if you could not cope with all the mandatory academic activities?", 0, 4, 0)
PSS5 = st.slider("In a semester, how often did you feel confident about your ability to handle your academic / university problems?", 0, 4, 0)
PSS9 = st.slider("In a semester, how often have you got angered due to bad performance or low grades that are beyond your control?", 0, 4, 0)
PSS10 = st.slider("In a semester, how often have you felt as if academic difficulties were piling up so high that you could not overcome them?", 0, 4, 0)

st.header("🧠 Depression (PHQ) Questionnaire")
st.markdown("Scale: 0 - Never, 1 - Almost Never, 2 - Sometimes, 3 - Fairly Often, 4 - Very Often")

PHQ2 = st.slider("In a semester, how often have you been feeling down, depressed or hopeless?", 0, 4, 0)
PHQ4 = st.slider("In a semester, how often have you felt tired or have little energy?", 0, 4, 0)
PHQ5 = st.slider("In a semester, how often have you had poor appetite or overeating?", 0, 4, 0)
PHQ6 = st.slider("In a semester, how often have you felt bad about yourself, or that you are a failure or have let yourself or your family down?", 0, 4, 0)
PHQ7 = st.slider("In a semester, how often have you been having trouble concentrating on things, such as reading books or watching television?", 0, 4, 0)
PHQ9 = st.slider("In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself?", 0, 4, 0)

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