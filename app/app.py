import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# PATH CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ml2" / "mir" / "logistic_regression.pkl"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "features2" / "mir" / "train.csv"

# SESSION STATE INITIALIZATION
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# LOAD MODEL
@st.cache_resource
def load_model_and_scaler():
    """Cache model and scaler to improve performance"""
    model = joblib.load(MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    
    features = [
        "PSS4", "PSS10", "PSS2", "PSS1", "PSS9", "PSS3", "PSS5",
        "PHQ2", "PHQ6", "PHQ4", "PHQ7", "PHQ9", "PHQ5"
    ]
    
    X_train = train_df[features]
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return model, scaler, features

try:
    model, scaler, FEATURES = load_model_and_scaler()
except Exception as e:
    st.error(f"❌ Error loading model or data: {str(e)}")
    st.stop()

# LABEL MAPPING
LABEL_MAP = {
    0: "Minimal",
    1: "Mild",
    2: "Moderate",
    3: "Moderately Severe",
    4: "Severe"
}

# COLOR SCHEME FOR SEVERITY LEVELS
SEVERITY_COLORS = {
    "Minimal": "#28a745",
    "Mild": "#90ee90",
    "Moderate": "#ffc107",
    "Moderately Severe": "#ff8c00",
    "Severe": "#dc3545"
}

# STREAMLIT UI CONFIG
st.set_page_config(
    page_title="Depression Screening System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Depression Screening System</h1>
        <p>Machine Learning Based Clinical Screening Tool</p>
    </div>
""", unsafe_allow_html=True)

# INFORMATION EXPANDER
with st.expander("ℹ️ About This Screening Tool", expanded=False):
    st.markdown("""
    **This screening tool uses:**
    - **PSS (Perceived Stress Scale)**: Measures your ability to cope with stress in daily life
    - **PHQ (Patient Health Questionnaire)**: Assesses depression symptoms
    
    **Important Notes:**
    - This is a research-based screening tool, NOT a medical diagnosis
    - Results should be discussed with a qualified healthcare professional
    - All responses are confidential and not stored
    
    **How to use:**
    1. Answer all questions honestly based on the past 2 weeks
    2. Click "Predict Depression Level" to see results
    3. Review the detailed analysis and recommendations
    """)

# PSS QUESTIONNAIRE
st.markdown('<div class="section-header"><h3>📋 Part 1: Perceived Stress Scale (PSS)</h3></div>', unsafe_allow_html=True)
st.caption("Please indicate how often you felt or thought in the following ways during the **past month**:")

PSS_QUESTIONS = {
    "PSS1": "How often have you been upset because of something that happened unexpectedly?",
    "PSS2": "How often have you felt that you were unable to control the important things in your life?",
    "PSS3": "How often have you felt nervous and stressed?",
    "PSS4": "How often have you felt that you could not cope with all the things that you had to do?",
    "PSS5": "How often have you felt confident about your ability to handle your personal problems?",
    "PSS9": "How often have you been angered because of things that happened that were outside of your control?",
    "PSS10": "How often have you felt difficulties were piling up so high that you could not overcome them?"
}

PSS_OPTIONS = ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"]

pss_inputs = {}
for i, (key, question) in enumerate(PSS_QUESTIONS.items(), 1):
    pss_inputs[key] = st.select_slider(
        f"**{i}. {question}**",
        options=range(5),
        format_func=lambda x: PSS_OPTIONS[x],
        key=f"pss_{key}"
    )

# PHQ QUESTIONNAIRE
st.markdown('<div class="section-header"><h3>📋 Part 2: Patient Health Questionnaire (PHQ)</h3></div>', unsafe_allow_html=True)
st.caption("Over the **last 2 weeks**, how often have you been bothered by the following problems:")

PHQ_QUESTIONS = {
    "PHQ2": "Feeling down, depressed or hopeless?",
    "PHQ4": "Feeling tired or having little energy?",
    "PHQ5": "Poor appetite or overeating?",
    "PHQ6": "Feeling bad about yourself, or that you are a failure or have let yourself or your family down?",
    "PHQ7": "Trouble concentrating on things, such as reading books or watching television?",
    "PHQ9": "Thoughts that you would be better off dead, or of hurting yourself?"
}

PHQ_OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]

phq_inputs = {}
for i, (key, question) in enumerate(PHQ_QUESTIONS.items(), 1):
    phq_inputs[key] = st.select_slider(
        f"**{i}. {question}**",
        options=range(4),
        format_func=lambda x: PHQ_OPTIONS[x],
        key=f"phq_{key}"
    )

# PREDICT BUTTON
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict Depression Level", type="primary")

# PREDICTION
if predict_button:
    st.session_state.prediction_made = True
    
    # Prepare input data
    user_data = {**pss_inputs, **phq_inputs}
    input_df = pd.DataFrame([user_data])[FEATURES]
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    predicted_label = LABEL_MAP[prediction]
    confidence = probabilities[prediction] * 100
    
    # Calculate total scores
    pss_total = sum(pss_inputs.values())
    phq_total = sum(phq_inputs.values())
    
    # RESULTS SECTION
    st.markdown("---")
    st.markdown("## 🧠 Screening Results")
    
    # Main result with colored background
    result_color = SEVERITY_COLORS[predicted_label]
    st.markdown(f"""
        <div style="background-color: {result_color}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="color: white; margin: 0;">Depression Level: {predicted_label}</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PSS Score (Stress)", f"{pss_total}/28", help="Range: 0-28 (Higher = More Stress)")
    with col2:
        st.metric("PHQ Score (Depression)", f"{phq_total}/18", help="Range: 0-18 (Higher = More Severe)")
    
    # Probability distribution chart
    st.markdown("### 📊 Probability Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(LABEL_MAP.values())
    colors = [SEVERITY_COLORS[label] for label in labels]
    
    bars = ax.barh(labels, probabilities, color=colors, alpha=0.7)
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_xlim(0, 1)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.01, i, f"{prob*100:.1f}%", va='center', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed breakdown table
    with st.expander("📈 View Detailed Probability Breakdown"):
        prob_df = pd.DataFrame({
            "Severity Level": labels,
            "Probability": [f"{p*100:.2f}%" for p in probabilities],
            "Raw Score": [f"{p:.4f}" for p in probabilities]
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = {
        "Minimal": """
        ✅ Your screening indicates minimal depression symptoms. Continue maintaining:
        - Regular sleep schedule
        - Physical activity and exercise
        - Social connections with friends and family
        - Healthy stress management techniques
        """,
        "Mild": """
        ⚠️ Your screening suggests mild depression symptoms. Consider:
        - Speaking with a counselor or therapist
        - Establishing a regular routine
        - Engaging in activities you enjoy
        - Reaching out to supportive friends or family
        - Monitoring your symptoms over the next few weeks
        """,
        "Moderate": """
        ⚠️ Your screening indicates moderate depression symptoms. We recommend:
        - Scheduling an appointment with a mental health professional
        - Talking to your academic advisor about support resources
        - Utilizing campus counseling services
        - Developing a self-care routine
        - Considering joining a support group
        """,
        "Moderately Severe": """
        🚨 Your screening suggests moderately severe depression. **Please take action:**
        - Contact a mental health professional as soon as possible
        - Reach out to campus crisis services or student health
        - Inform a trusted friend or family member
        - Consider temporary academic accommodations if needed
        - Prioritize your mental health above other obligations
        """,
        "Severe": """
        🆘 Your screening indicates severe depression symptoms. **Immediate action needed:**
        - **Call Kaan Pete Roi: +880 2 5086 5486 (24/7) immediately**
        - Contact National Institute of Mental Health: +880 2 9004511
        - Visit your campus health center or nearest emergency services
        - Do NOT isolate yourself - reach out for help now
        - Inform someone you trust about how you're feeling
        - Consider visiting NIMH or BSMMU Psychiatry Department
        
        **Bangladesh Crisis Resources:**
        - **Kaan Pete Roi (কান পেতে রই)**: +880 2 5086 5486 (24/7)
        - **Shuni (শুনি)**: 09678 224466 (8 AM - 11 PM)
        - **Emergency Services**: 999
        """
    }
    
    st.info(recommendations[predicted_label])
    
    # Resources section
    with st.expander("📞 Mental Health Resources (Bangladesh)"):
        st.markdown("""
        **🇧🇩 Bangladesh Crisis Support:**
        - **Kaan Pete Roi (কান পেতে রই)**: **+880 9612-119911** (24/7 Helpline)
        - **Moner Bondhu Foundation**: **+880 1776632344** (Mental Health Support)
        
        **Bangladesh Healthcare Facilities:**
        - National Institute of Mental Health & Hospital, Sher-e-Bangla Nagar, Dhaka
        - Bangabandhu Sheikh Mujib Medical University (BSMMU) - Psychiatry Dept.
        - Dhaka Medical College Hospital - Mental Health Unit
        - Bangladesh Association for Psychiatry (BAP)
        
        **University Resources in Bangladesh:**
        - University Health Center/Medical Center
        - Student Counseling Services
        - Proctor's Office/Student Welfare
        - Department of Psychology (if available)
        
        **Online & NGO Resources:**
        - **Moner School**: Online mental health awareness platform
        - **Aanonda Foundation**: Youth mental health support
        - **Antara Counselling Center**: Professional counseling services
        - **BRAC**: Community health programs
        
        **International Resources:**
        - International Association for Suicide Prevention: [www.iasp.info](https://www.iasp.info)
        - WHO Mental Health Resources: [www.who.int/mental_health](https://www.who.int/mental_health)
        
        **Emergency:**
        - National Emergency Service: **999**
        """)

# FOOTER
st.markdown("---")
st.warning("""
⚠️ **Important Disclaimer**

This system is a research-based screening tool and **NOT a medical diagnosis**. Results should be interpreted by a qualified healthcare professional. If you experience distress, harmful thoughts, or crisis, please seek professional support immediately.

This tool uses machine learning for educational and screening purposes only. Always consult with licensed mental health professionals for proper diagnosis and treatment.
""")

st.caption("🔒 Your responses are not stored or shared. All data remains confidential during this session.")

# Reset button
if st.session_state.prediction_made:
    if st.button("🔄 Take Assessment Again"):
        st.session_state.prediction_made = False
        st.rerun()