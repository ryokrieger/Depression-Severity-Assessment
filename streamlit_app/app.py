import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

# PAGE CONFIG
st.set_page_config(page_title="Depression Assessment System", layout="centered")

st.title("🧠 AI-Based Depression Assessment System")
st.write("Structured Machine Learning Deployment with LIME Explainability")

# SAFE PATH HANDLING (CLOUD + LOCAL)
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "logistic_regression.pkl"
DATA_PATH = BASE_DIR / "train.csv"

# LOAD MODEL & TRAINING DATA
if not MODEL_PATH.exists():
    st.error("❌ logistic_regression.pkl not found in streamlit_app/")
    st.stop()

if not DATA_PATH.exists():
    st.error("❌ train.csv not found in streamlit_app/")
    st.stop()

model = joblib.load(MODEL_PATH)
train_df = pd.read_csv(DATA_PATH)

# ✅ Selected MIR Features (CRITICAL ORDER)
feature_cols = [
    "PSS4", "PSS10", "PSS2", "PSS1", "PSS9",
    "PSS3", "PSS5",
    "PHQ2", "PHQ6", "PHQ4", "PHQ7", "PHQ9", "PHQ5"
]

X_train = train_df[feature_cols].values
y_train = train_df["DepressionEncoded"].values

# LIME EXPLAINER (AIDA STYLE)
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_cols,
    class_names=["Not Depressed", "Depressed"],
    mode="classification",
    discretize_continuous=True
)

# QUESTIONNAIRE UI
st.subheader("📝 Stress (PSS) Questionnaire")

PSS1 = st.slider("PSS1: Upset due to academic affairs", 0, 4, 2)
PSS2 = st.slider("PSS2: Unable to control important academic matters", 0, 4, 2)
PSS3 = st.slider("PSS3: Nervous and stressed due to academic pressure", 0, 4, 2)
PSS4 = st.slider("PSS4: Could not cope with academic activities", 0, 4, 2)
PSS5 = st.slider("PSS5: Confident in handling academic problems", 0, 4, 2)
PSS6 = st.slider("PSS6: Academic life going your way", 0, 4, 2)
PSS7 = st.slider("PSS7: Able to control academic irritations", 0, 4, 2)
PSS8 = st.slider("PSS8: Academic performance on top", 0, 4, 2)
PSS9 = st.slider("PSS9: Angered due to bad performance", 0, 4, 2)
PSS10 = st.slider("PSS10: Academic difficulties piling up", 0, 4, 2)

st.subheader("📝 Depression (PHQ) Questionnaire")

PHQ1 = st.slider("PHQ1: Little interest or pleasure", 0, 4, 2)
PHQ2 = st.slider("PHQ2: Feeling down or hopeless", 0, 4, 2)
PHQ3 = st.slider("PHQ3: Sleep problems", 0, 4, 2)
PHQ4 = st.slider("PHQ4: Feeling tired", 0, 4, 2)
PHQ5 = st.slider("PHQ5: Appetite problems", 0, 4, 2)
PHQ6 = st.slider("PHQ6: Feeling like a failure", 0, 4, 2)
PHQ7 = st.slider("PHQ7: Trouble concentrating", 0, 4, 2)
PHQ8 = st.slider("PHQ8: Moving/speaking unusually", 0, 4, 2)
PHQ9 = st.slider("PHQ9: Thoughts of self-harm", 0, 4, 2)

# MATCH FEATURE ORDER EXACTLY
user_input = np.array([[
    PSS4, PSS10, PSS2, PSS1, PSS9,
    PSS3, PSS5,
    PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
]])

# PREDICTION
st.markdown("---")
if st.button("🔍 Predict Depression"):

    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0]

    label = "Depressed" if pred == 1 else "Not Depressed"

    st.subheader("✅ Prediction Result")
    st.success(f"### 🧠 Prediction: {label}")
    st.write(f"**Probability (Not Depressed):** {prob[0]:.4f}")
    st.write(f"**Probability (Depressed):** {prob[1]:.4f}")


    # LIME EXPLANATION

    st.markdown("## 🔍 LIME Explainability")

    exp = explainer.explain_instance(
        data_row=user_input[0],
        predict_fn=model.predict_proba,
        num_features=10
    )

    lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"])
    st.dataframe(lime_df)

    st.markdown("🔹 **Positive Impact → Increases Depression Risk**")
    st.markdown("🔹 **Negative Impact → Decreases Depression Risk**")

    # Save LIME Explanation
    exp.save_to_file("lime_explanation.html")
    st.info("📄 LIME explanation saved as `lime_explanation.html`")

# FOOTER
st.markdown("---")
st.caption("AI-Based Depression Assessment System")