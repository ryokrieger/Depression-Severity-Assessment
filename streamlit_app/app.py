import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# CONFIG
st.set_page_config(
    page_title="Depression Prediction System",
    layout="centered",
    page_icon="🧠"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ml2" / "mir" / "logistic_regression.pkl"

# LOAD MODEL
@st.cache_resource
def load_model():
    """Load model with caching for better performance"""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

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

# COLOR MAPPING for results
COLOR_MAP = {
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴",
    4: "🔴"
}

# STREAMLIT UI
st.title("🧠 AI-Based Depression Assessment System")
st.markdown("""
This system predicts **Depression Level** based on validated **PSS (Perceived Stress Scale)** 
and **PHQ (Patient Health Questionnaire)** assessments.

⚠️ **Important:** This tool is for educational/screening purposes only and does not replace 
professional mental health evaluation.
""")

# Add tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Assessment", "ℹ️ About", "🆘 Resources"])

with tab1:
    st.header("📊 Stress (PSS) Questionnaire")
    st.markdown("**Scale:** 0 = Never, 1 = Almost Never, 2 = Sometimes, 3 = Fairly Often, 4 = Very Often")
    
    with st.expander("PSS Questions", expanded=True):
        PSS1 = st.slider(
            "Q1: In a semester, how often have you felt upset due to something that happened in your academic affairs?",
            0, 4, 0, key="pss1"
        )
        PSS2 = st.slider(
            "Q2: How often did you feel unable to control important things in your academic affairs?",
            0, 4, 0, key="pss2"
        )
        PSS3 = st.slider(
            "Q3: How often did you feel nervous and stressed because of academic pressure?",
            0, 4, 0, key="pss3"
        )
        PSS4 = st.slider(
            "Q4: How often did you feel you could not cope with all mandatory academic activities?",
            0, 4, 0, key="pss4"
        )
        PSS5 = st.slider(
            "Q5: How often did you feel confident about your ability to handle academic/university problems?",
            0, 4, 0, key="pss5"
        )
        PSS9 = st.slider(
            "Q6: How often have you gotten angered due to bad performance or low grades beyond your control?",
            0, 4, 0, key="pss9"
        )
        PSS10 = st.slider(
            "Q7: How often have you felt academic difficulties were piling up so high you couldn't overcome them?",
            0, 4, 0, key="pss10"
        )

    st.header("🧠 Depression (PHQ) Questionnaire")
    st.markdown("**Scale:** 0 = Never, 1 = Almost Never, 2 = Sometimes, 3 = Fairly Often, 4 = Very Often")
    
    with st.expander("PHQ Questions", expanded=True):
        PHQ2 = st.slider(
            "Q1: How often have you been feeling down, depressed or hopeless?",
            0, 4, 0, key="phq2"
        )
        PHQ4 = st.slider(
            "Q2: How often have you felt tired or have little energy?",
            0, 4, 0, key="phq4"
        )
        PHQ5 = st.slider(
            "Q3: How often have you had poor appetite or overeating?",
            0, 4, 0, key="phq5"
        )
        PHQ6 = st.slider(
            "Q4: How often have you felt bad about yourself, or that you are a failure?",
            0, 4, 0, key="phq6"
        )
        PHQ7 = st.slider(
            "Q5: How often have you had trouble concentrating on things?",
            0, 4, 0, key="phq7"
        )
        PHQ9 = st.slider(
            "Q6: How often have you had thoughts that you would be better off dead, or of hurting yourself?",
            0, 4, 0, key="phq9",
            help="If you're experiencing these thoughts, please seek immediate professional help."
        )

    # Calculate scores
    pss_score = PSS1 + PSS2 + PSS3 + PSS4 + PSS9 + PSS10 + (4 - PSS5)  # PSS5 is reverse scored
    phq_score = PHQ2 + PHQ4 + PHQ5 + PHQ6 + PHQ7 + PHQ9

    col1, col2 = st.columns(2)
    with col1:
        st.metric("PSS Score", f"{pss_score}/28")
    with col2:
        st.metric("PHQ Score", f"{phq_score}/24")

    # PREDICTION
    if st.button("🔍 Predict Depression Level", type="primary", use_container_width=True):
        
        input_data = pd.DataFrame([[
            PSS4, PSS10, PSS2, PSS1, PSS9, PSS3, PSS5,
            PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
        ]], columns=FEATURE_COLS)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.divider()
        st.subheader("✅ Assessment Result")
        
        result_color = COLOR_MAP[prediction]
        st.success(f"{result_color} **Predicted Level: {LABEL_MAP[prediction]}**")
        
        # Add interpretation
        interpretations = {
            0: "Your responses suggest minimal or no depression symptoms. Continue monitoring your mental health.",
            1: "Your responses suggest mild depression symptoms. Consider speaking with a counselor or mental health professional.",
            2: "Your responses suggest moderate depression symptoms. We recommend consulting with a mental health professional.",
            3: "Your responses suggest moderately severe depression. Please seek professional mental health support soon.",
            4: "Your responses suggest severe depression. Please seek immediate professional help. Contact a crisis helpline if needed."
        }
        
        st.info(interpretations[prediction])

        prob_df = pd.DataFrame({
            "Depression Level": list(LABEL_MAP.values()),
            "Probability (%)": [p * 100 for p in probabilities]
        })

        st.subheader("📈 Prediction Confidence Distribution")
        st.bar_chart(prob_df.set_index("Depression Level"))
        
        # Show detailed probabilities
        with st.expander("View Detailed Probabilities"):
            st.dataframe(prob_df, use_container_width=True)
        
        # Crisis warning for severe cases
        if prediction >= 3 or PHQ9 >= 2:
            st.error("""
            ⚠️ **Important Notice:** If you're experiencing thoughts of self-harm or suicide, 
            please seek immediate help from a mental health professional or crisis helpline.
            """)

with tab2:
    st.header("About This System")
    st.markdown("""
    ### Assessment Scales Used
    
    **PSS (Perceived Stress Scale):**
    - Measures stress levels in academic settings
    - Score range: 0-28
    - Higher scores indicate higher stress
    
    **PHQ (Patient Health Questionnaire):**
    - Screens for depression symptoms
    - Score range: 0-24
    - Based on DSM-5 criteria
    
    ### Model Information
    - **Algorithm:** Logistic Regression
    - **Purpose:** Educational screening tool
    - **Limitation:** Not a diagnostic tool
    
    ### Disclaimer
    This system provides preliminary assessments only. For accurate diagnosis and treatment, 
    please consult qualified mental health professionals.
    """)

with tab3:
    st.header("Mental Health Resources")
    st.markdown("""
    ### 🆘 Crisis Support
    
    **International:**
    - 🌐 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
    
    **United States:**
    - 📞 988 Suicide & Crisis Lifeline: Call/Text 988
    - 💬 Crisis Text Line: Text "HELLO" to 741741
    
    **United Kingdom:**
    - 📞 Samaritans: 116 123
    
    **Bangladesh:**
    - 📞 Kaan Pete Roi (Emotional Support): 09612-119911
    
    ### 🏥 Professional Help
    - Contact your university counseling center
    - Speak with your primary care physician
    - Find a licensed therapist or psychiatrist
    
    ### 📚 Self-Care Resources
    - Mental Health America: https://mhanational.org/
    - National Alliance on Mental Illness: https://www.nami.org/
    """)

# Footer
st.divider()
st.caption("⚕️ Developed for educational purposes | Always consult healthcare professionals for diagnosis")