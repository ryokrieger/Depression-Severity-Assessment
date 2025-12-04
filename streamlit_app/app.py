import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# CONFIG
st.set_page_config(page_title="AI Depression Detection System", layout="centered")

MODEL_PATH = Path("../models/ml2/mir/logistic_regression.pkl")
DATA_PATH = Path("../data/processed/features2/mir/train.csv")

# LOAD MODEL & DATA
model = joblib.load(MODEL_PATH)
train_df = pd.read_csv(DATA_PATH)

X_train = train_df.drop(columns=["DepressionEncoded"])

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

# LIME EXPLAINER
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=FEATURE_COLS,
    class_names=list(LABEL_MAP.values()),
    mode="classification"
)

# STREAMLIT UI
st.title("🎓 AI-Based Depression Assessment System")
st.markdown("Real-time **Depression Detection + Explainable AI (LIME)**")

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

# PREDICTION + LIME
if st.button("🔍 Predict & Explain"):

    input_df = pd.DataFrame([[
        PSS4, PSS10, PSS2, PSS1, PSS9, PSS3, PSS5,
        PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
    ]], columns=FEATURE_COLS)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Depression Level: **{LABEL_MAP[prediction]}**")

    prob_df = pd.DataFrame({
        "Depression Level": list(LABEL_MAP.values()),
        "Probability": probabilities
    })

    st.subheader("📈 Prediction Probability Distribution")
    st.bar_chart(prob_df.set_index("Depression Level"))

    # LIME EXPLANATION
    st.subheader("🧠 LIME Explanation (Why This Prediction?)")

    exp = explainer.explain_instance(
        data_row=input_df.values[0],
        predict_fn=model.predict_proba,
        num_features=10
    )

    lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"])

    st.dataframe(lime_df)

    fig, ax = plt.subplots()
    colors = ["green" if x > 0 else "red" for x in lime_df["Impact"]]
    ax.barh(lime_df["Feature"], lime_df["Impact"])
    ax.set_title("Top Contributing Features (LIME)")
    st.pyplot(fig)