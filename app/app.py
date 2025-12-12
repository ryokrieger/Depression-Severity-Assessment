import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "03 - Third Working" / "Machine Learning" / "mir" / "logistic_regression.pkl"
SCALER_PATH = BASE_DIR / "features" / "03 - Third Working" / "mir" / "scaler.pkl"

# LOAD MODEL & SCALER
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_models()

# CLASS LABELS & COLORS
class_names = {
    0: "Minimal",
    1: "Mild",
    2: "Moderate",
    3: "Moderately Severe",
    4: "Severe"
}

class_colors = {
    0: "#28a745",
    1: "#6f42c1",
    2: "#ffc107",
    3: "#fd7e14",
    4: "#dc3545"
}

# QUESTIONS WITH RESPONSE OPTIONS
stress_questions = [
    {
        "key": "Q1",
        "text": "How often have you felt unable to handle your personal problems?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS4"
    },
    {
        "key": "Q2",
        "text": "How often have you felt difficulties were piling up so high that you could not overcome them?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS10"
    },
    {
        "key": "Q3",
        "text": "How often have you felt that you were unable to control the important things in your life?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS2"
    },
    {
        "key": "Q4",
        "text": "How often have you been upset because of something that happened unexpectedly?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS1"
    },
    {
        "key": "Q5",
        "text": "How often have you been angered because of things outside of your control?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS9"
    },
    {
        "key": "Q6",
        "text": "How often have you felt nervous and stressed?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS3"
    },
    {
        "key": "Q7",
        "text": "How often have you felt that things were not going your way?",
        "options": ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"],
        "feature": "PSS5"
    }
]

depression_questions = [
    {
        "key": "Q8",
        "text": "How often have you felt down, sad, or hopeless?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ2"
    },
    {
        "key": "Q9",
        "text": "How often have you felt bad about yourself or like you've let others down?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ6"
    },
    {
        "key": "Q10",
        "text": "How often have you felt tired or had little energy?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ4"
    },
    {
        "key": "Q11",
        "text": "How often have you had trouble concentrating on everyday tasks?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ7"
    },
    {
        "key": "Q12",
        "text": "How often have you had thoughts of death or self-harm?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ9"
    },
    {
        "key": "Q13",
        "text": "How often have you had poor appetite or been overeating?",
        "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        "feature": "PHQ5"
    }
]

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Depression Severity Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .question-text {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=150)
    st.title("About This Tool")
    st.info("""
    This AI-powered system assesses mental health and predicts depression severity 
    using validated clinical questionnaires.
    
    The model uses machine learning trained on clinical data to provide accurate assessments.
    """)
    
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    - Answer all questions honestly
    - Think about the **past month** for stress-related questions
    - Think about the **past 2 weeks** for mood-related questions
    - Select the response that best describes your experience
    """)
    
    st.markdown("---")
    show_feature_analysis = st.checkbox("Show detailed analysis", value=False)

# MAIN CONTENT
st.markdown('<div class="main-header">🧠 Depression Severity Assessment</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>Welcome!</b> Please answer the following questions based on your recent experiences.
    This assessment typically takes 3-5 minutes to complete. All responses are confidential.
</div>
""", unsafe_allow_html=True)

# Initialize session state for responses
if 'responses' not in st.session_state:
    st.session_state.responses = {}

# CREATE TWO COLUMNS
col1, col2 = st.columns(2)

# STRESS-RELATED QUESTIONS (LEFT COLUMN)
with col1:
    st.markdown("### 🟡 Recent Life Experiences")
    st.markdown("*Thinking about the past month...*")
    
    for q in stress_questions:
        st.markdown(f'<div class="question-text">{q["text"]}</div>', unsafe_allow_html=True)
        response = st.select_slider(
            label=f"Response for {q['key']}",
            options=q["options"],
            key=q["key"],
            label_visibility="collapsed"
        )
        st.session_state.responses[q["feature"]] = q["options"].index(response)
        st.markdown("<br>", unsafe_allow_html=True)

# MOOD-RELATED QUESTIONS (RIGHT COLUMN)
with col2:
    st.markdown("### 🔵 Recent Mood & Wellbeing")
    st.markdown("*Over the past 2 weeks, how often have you been bothered by:*")
    
    for q in depression_questions:
        st.markdown(f'<div class="question-text">{q["text"]}</div>', unsafe_allow_html=True)
        response = st.select_slider(
            label=f"Response for {q['key']}",
            options=q["options"],
            key=q["key"],
            label_visibility="collapsed"
        )
        st.session_state.responses[q["feature"]] = q["options"].index(response)
        st.markdown("<br>", unsafe_allow_html=True)

# FEATURE VECTOR (in correct order)
input_features = np.array([[
    st.session_state.responses.get("PSS4", 0),
    st.session_state.responses.get("PSS10", 0),
    st.session_state.responses.get("PSS2", 0),
    st.session_state.responses.get("PSS1", 0),
    st.session_state.responses.get("PSS9", 0),
    st.session_state.responses.get("PSS3", 0),
    st.session_state.responses.get("PSS5", 0),
    st.session_state.responses.get("PHQ2", 0),
    st.session_state.responses.get("PHQ6", 0),
    st.session_state.responses.get("PHQ4", 0),
    st.session_state.responses.get("PHQ7", 0),
    st.session_state.responses.get("PHQ9", 0),
    st.session_state.responses.get("PHQ5", 0)
]])

# Calculate totals
stress_total = sum([st.session_state.responses.get(q["feature"], 0) for q in stress_questions])
mood_total = sum([st.session_state.responses.get(q["feature"], 0) for q in depression_questions])

# PREDICTION BUTTON
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("🔍 Analyze Depression Severity", use_container_width=True, type="primary")

if predict_button:
    # Check if all questions are answered
    if len(st.session_state.responses) < 13:
        st.warning("⚠️ Please answer all questions before proceeding with the analysis.")
    else:
        with st.spinner("Analyzing responses..."):
            input_scaled = scaler.transform(input_features)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            predicted_label = class_names[prediction]
            confidence = probabilities[prediction] * 100
            
            # MAIN PREDICTION RESULT
            st.markdown("---")
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: {class_colors[prediction]};">Predicted Depression Level: {predicted_label}</h2>
                <p style="font-size: 1.2rem;">Confidence Level: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # VISUALIZATION SECTION
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("📊 Severity Probability Distribution")
                
                # Create DataFrame for better visualization
                prob_df = pd.DataFrame({
                    'Severity': [class_names[i] for i in range(5)],
                    'Probability': probabilities
                })
                
                # Matplotlib bar chart with colors
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = [class_colors[i] for i in range(5)]
                bars = ax.barh(prob_df['Severity'], prob_df['Probability'], color=colors, alpha=0.7)
                
                # Highlight prediction
                bars[prediction].set_alpha(1.0)
                bars[prediction].set_edgecolor('black')
                bars[prediction].set_linewidth(2)
                
                ax.set_xlabel('Probability', fontsize=12)
                ax.set_xlim(0, 1)
                ax.grid(axis='x', alpha=0.3)
                
                for i, (sev, prob) in enumerate(zip(prob_df['Severity'], prob_df['Probability'])):
                    ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with viz_col2:
                st.subheader("📈 Response Summary")
                
                # Score comparison chart
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                
                categories = ['Stress\nResponses', 'Mood\nResponses']
                scores = [stress_total, mood_total]
                max_scores = [28, 18]
                percentages = [(s/m)*100 for s, m in zip(scores, max_scores)]
                
                bars = ax2.bar(categories, percentages, color=['#ffc107', '#1f77b4'], alpha=0.7)
                
                for bar, score, max_score in zip(bars, scores, max_scores):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{score}/{max_score}',
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax2.set_ylabel('Percentage of Maximum Score', fontsize=12)
                ax2.set_ylim(0, 110)
                ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # DETAILED PROBABILITIES
            st.markdown("---")
            st.subheader("🔢 Detailed Probability Breakdown")
            
            prob_cols = st.columns(5)
            for i, (col, prob) in enumerate(zip(prob_cols, probabilities)):
                with col:
                    st.metric(
                        label=class_names[i],
                        value=f"{prob:.1%}",
                        delta="Predicted" if i == prediction else None
                    )
            
            # DETAILED ANALYSIS (if checkbox enabled)
            if show_feature_analysis:
                st.markdown("---")
                st.subheader("🎯 Response Pattern Analysis")
                
                feature_names = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
                               'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']
                feature_values = input_features[0]
                
                # Normalize feature values for visualization
                max_values = [4,4,4,4,4,4,4,3,3,3,3,3,3]
                feature_importance = feature_values / np.array(max_values)
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                colors_feat = ['#ffc107' if i < 7 else '#1f77b4' for i in range(13)]
                
                bars = ax3.barh(feature_names, feature_importance, color=colors_feat, alpha=0.7)
                ax3.set_xlabel('Normalized Response Score (0-1)', fontsize=12)
                ax3.set_title('Individual Question Response Intensity', fontsize=14, fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
                
                # Add actual values as text
                for i, (bar, val) in enumerate(zip(bars, feature_values)):
                    ax3.text(bar.get_width() + 0.02, i, f'{int(val)}', 
                            va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            # RECOMMENDATIONS
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 0:
                st.success("""
                **Minimal Depression**: Your responses suggest minimal depressive symptoms.
                Continue maintaining healthy habits and stress management practices.
                """)
            elif prediction == 1:
                st.info("""
                **Mild Depression**: Consider lifestyle modifications such as regular exercise,
                adequate sleep, and stress reduction techniques. Monitor your symptoms.
                """)
            elif prediction == 2:
                st.warning("""
                **Moderate Depression**: It's advisable to consult with a mental health professional.
                Consider therapy and maintain a support network.
                """)
            elif prediction == 3:
                st.error("""
                **Moderately Severe Depression**: Professional help is strongly recommended.
                Contact a mental health provider for assessment and treatment options.
                """)
            else:
                st.error("""
                **Severe Depression**: Immediate professional intervention is recommended.
                Please reach out to a mental health crisis service or healthcare provider.
                """)
            
            # CRISIS RESOURCES
            if prediction >= 3 or st.session_state.responses.get("PHQ9", 0) >= 2:
                st.markdown("---")
                st.error("""
                **🆘 Crisis Resources (Bangladesh):**
                - **Kaan Pete Roi (KPR)**: +88 09612 119911 (Emotional support helpline, 24/7)
                - **Moner Bondhu**: +88 01776 632344 (Mental health support)
                - **National Mental Health Institute**: +8802-223374409
                - **Emergency Services**: 999 (National Emergency)
                
                **International Resources:**
                - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
                """)

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>⚠️ Important Disclaimer</b></p>
    <p>This tool is for educational and research purposes only. It does NOT provide medical diagnosis or treatment.
    Always consult qualified healthcare professionals for mental health concerns.</p>
    <p><i>Assessment Time: {}</i></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)