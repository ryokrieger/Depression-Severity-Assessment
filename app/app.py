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
MODEL_PATH = BASE_DIR / "models" / "ml2" / "mir" / "logistic_regression.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "features2" / "mir" / "scaler.pkl"

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

# QUESTION DESCRIPTIONS
pss_questions = {
    "PSS1": "How often have you been upset because of something that happened unexpectedly?",
    "PSS2": "How often have you felt that you were unable to control the important things in your life?",
    "PSS3": "How often have you felt nervous and stressed?",
    "PSS4": "How often have you felt unable to handle your personal problems?",
    "PSS5": "How often have you felt that things were not going your way?",
    "PSS9": "How often have you been angered because of things outside of your control?",
    "PSS10": "How often have you felt difficulties were piling up so high that you could not overcome them?"
}

phq_questions = {
    "PHQ2": "How often have you felt down, sad, or hopeless?",
    "PHQ4": "How often have you felt tired or had little energy?",
    "PHQ5": "How often have you had poor appetite or been overeating?",
    "PHQ6": "How often have you felt bad about yourself or like you've let others down?",
    "PHQ7": "How often have you had trouble concentrating on everyday tasks?",
    "PHQ9": "How often have you had thoughts of death or self-harm?"
}

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Depression Severity Prediction",
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
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=150)
    st.title("About This Tool")
    st.info("""
    This AI-powered system predicts depression severity using:
    
    - **PSS (Perceived Stress Scale)**: Measures stress levels
    - **PHQ (Patient Health Questionnaire)**: Assesses depressive symptoms
    
    The model uses logistic regression trained on validated clinical data.
    """)
    
    st.markdown("---")
    st.subheader("Response Scale Guide")
    
    with st.expander("PSS Scale (0-4)"):
        st.markdown("""
        - **0**: Never
        - **1**: Almost Never
        - **2**: Sometimes
        - **3**: Fairly Often
        - **4**: Very Often
        """)
    
    with st.expander("PHQ Scale (0-3)"):
        st.markdown("""
        - **0**: Not at all
        - **1**: Several days
        - **2**: More than half the days
        - **3**: Nearly every day
        """)
    
    st.markdown("---")
    show_explanations = st.checkbox("Show question explanations", value=False)
    show_feature_importance = st.checkbox("Show feature analysis", value=False)

# MAIN CONTENT
st.markdown('<div class="main-header">🧠 Depression Severity Assessment</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>Instructions:</b> Please answer the following questions based on your experiences over the past month.
    This assessment typically takes 3-5 minutes to complete.
</div>
""", unsafe_allow_html=True)

# CREATE TWO COLUMNS
col1, col2 = st.columns(2)

# STRESS (PSS) INPUTS
with col1:
    st.markdown("### 🟡 Perceived Stress Scale (PSS)")
    st.markdown("*How often have you experienced the following in the past month?*")
    
    PSS4 = st.slider(
        "Not confident on handling problems",
        0, 4, 0,
        help=pss_questions["PSS4"] if show_explanations else None
    )
    PSS10 = st.slider(
        "Difficulties piling up",
        0, 4, 0,
        help=pss_questions["PSS10"] if show_explanations else None
    )
    PSS2 = st.slider(
        "Unable to control important things",
        0, 4, 0,
        help=pss_questions["PSS2"] if show_explanations else None
    )
    PSS1 = st.slider(
        "Upset by unexpected events",
        0, 4, 0,
        help=pss_questions["PSS1"] if show_explanations else None
    )
    PSS9 = st.slider(
        "Angered by uncontrollable events",
        0, 4, 0,
        help=pss_questions["PSS9"] if show_explanations else None
    )
    PSS3 = st.slider(
        "Feeling nervous and stressed",
        0, 4, 0,
        help=pss_questions["PSS3"] if show_explanations else None
    )
    PSS5 = st.slider(
        "Things not going your way",
        0, 4, 0,
        help=pss_questions["PSS5"] if show_explanations else None
    )
    
    pss_total = PSS4 + PSS10 + PSS2 + PSS1 + PSS9 + PSS3 + PSS5
    st.metric("Total PSS Score", pss_total, help="Range: 0-28")

# DEPRESSION (PHQ) INPUTS
with col2:
    st.markdown("### 🔵 Patient Health Questionnaire (PHQ)")
    st.markdown("*Over the past 2 weeks, how often have you been bothered by:*")
    
    PHQ2 = st.slider(
        "Feeling down or depressed",
        0, 3, 0,
        help=phq_questions["PHQ2"] if show_explanations else None
    )
    PHQ6 = st.slider(
        "Feeling like a failure",
        0, 3, 0,
        help=phq_questions["PHQ6"] if show_explanations else None
    )
    PHQ4 = st.slider(
        "Feeling tired or low energy",
        0, 3, 0,
        help=phq_questions["PHQ4"] if show_explanations else None
    )
    PHQ7 = st.slider(
        "Trouble concentrating",
        0, 3, 0,
        help=phq_questions["PHQ7"] if show_explanations else None
    )
    PHQ9 = st.slider(
        "Thoughts of self-harm",
        0, 3, 0,
        help=phq_questions["PHQ9"] if show_explanations else None
    )
    PHQ5 = st.slider(
        "Poor appetite or overeating",
        0, 3, 0,
        help=phq_questions["PHQ5"] if show_explanations else None
    )
    
    phq_total = PHQ2 + PHQ6 + PHQ4 + PHQ7 + PHQ9 + PHQ5
    st.metric("Total PHQ Score", phq_total, help="Range: 0-18")

# FEATURE VECTOR
input_features = np.array([[
    PSS4, PSS10, PSS2, PSS1, PSS9, PSS3, PSS5,
    PHQ2, PHQ6, PHQ4, PHQ7, PHQ9, PHQ5
]])

# PREDICTION BUTTON
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("🔍 Analyze Depression Severity", use_container_width=True, type="primary")

if predict_button:
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
            <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VISUALIZATION SECTION
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("📊 Probability Distribution")
            
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
            st.subheader("📈 Score Breakdown")
            
            # Score comparison chart
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            
            categories = ['PSS\nStress', 'PHQ\nDepression']
            scores = [pss_total, phq_total]
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
        
        # FEATURE IMPORTANCE (if checkbox enabled)
        if show_feature_importance:
            st.markdown("---")
            st.subheader("🎯 Feature Contribution Analysis")
            
            feature_names = ['PSS4', 'PSS10', 'PSS2', 'PSS1', 'PSS9', 'PSS3', 'PSS5',
                           'PHQ2', 'PHQ6', 'PHQ4', 'PHQ7', 'PHQ9', 'PHQ5']
            feature_values = input_features[0]
            
            # Normalize feature values for visualization
            feature_importance = feature_values / np.array([4,4,4,4,4,4,4,3,3,3,3,3,3])
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            colors_feat = ['#ffc107' if 'PSS' in name else '#1f77b4' for name in feature_names]
            
            bars = ax3.barh(feature_names, feature_importance, color=colors_feat, alpha=0.7)
            ax3.set_xlabel('Normalized Score (0-1)', fontsize=12)
            ax3.set_title('Individual Question Scores', fontsize=14, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
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
        if prediction >= 3 or PHQ9 >= 2:
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
    <p><i>Timestamp: {}</i></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)