"""
Streamlit web app for Saudi Tech Role Predictor.

Run from the app folder:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "role_model.pkl"
VECTORIZER_PATH = APP_DIR / "tfidf_vectorizer.pkl"

ROLE_DISPLAY_NAMES = {
    "ai/ml engineer": "AI/ML Engineer",
    "data analyst": "Data Analyst",
    "data engineer": "Data Engineer",
    "data scientist": "Data Scientist",
}


st.set_page_config(
    page_title="Saudi Tech Role Predictor",
    page_icon="📊",
    layout="centered",
)


CUSTOM_CSS = """
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #14532d 100%);
    padding: 2rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero-card h1 {
    margin-bottom: 0.3rem;
}
.info-card {
    background: white;
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.result-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 20px;
    border: 1px solid #dbeafe;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    margin-top: 1rem;
}
.small-note {
    color: #64748b;
    font-size: 0.92rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    """Load trained model and TF-IDF vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = []
        if not MODEL_PATH.exists():
            missing.append(str(MODEL_PATH.name))
        if not VECTORIZER_PATH.exists():
            missing.append(str(VECTORIZER_PATH.name))

        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Run `python train_and_save_model.py` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def format_role(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(str(role).lower(), str(role).title())


def get_prediction_scores(model, vectorized_input):
    """
    Return recommended class and matching scores.

    RandomForestClassifier supports predict_proba, so confidence scores are available.
    The fallback handles models without predict_proba.
    """
    predicted_role = model.predict(vectorized_input)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        classes = model.classes_
        scores = {
            str(role): float(prob) * 100
            for role, prob in zip(classes, probabilities)
        }
        confidence = scores[str(predicted_role)]
        return predicted_role, confidence, scores

    return predicted_role, None, {str(predicted_role): 100.0}


st.markdown(
    """
    <div class="hero-card">
        <h1>Saudi Tech Role Predictor</h1>
        <p>
            A machine learning web application that recommends the most suitable
            Saudi AI/Data job role based on user-entered skills.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <b>How it works:</b><br>
        The model compares your entered skills with patterns learned from Saudi
        technology job postings. Your text is transformed using TF-IDF and then
        passed to a trained classification model.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_assets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Open the terminal inside the app folder and run: `python train_and_save_model.py`")
    st.stop()
except Exception as exc:
    st.error("The model or vectorizer could not be loaded.")
    st.exception(exc)
    st.stop()


with st.form("prediction_form"):
    user_input = st.text_area(
        "Enter your skills",
        placeholder="Python, SQL, Machine Learning, Power BI, Data Visualization, Statistics",
        height=150,
    )

    submitted = st.form_submit_button("Predict Role")

if submitted:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.warning("Please enter at least a few skills before predicting.")
        st.stop()

    try:
        vectorized_input = vectorizer.transform([cleaned_input])
        recommended_role, confidence, scores = get_prediction_scores(model, vectorized_input)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Recommended Role")
        st.success(format_role(recommended_role))

        if confidence is not None:
            st.metric("Confidence Score", f"{confidence:.1f}%")
        else:
            st.info("This model does not support probability scores.")

        st.write("### Role Matching Scores")

        scores_df = (
            pd.DataFrame(
                {
                    "Role": [format_role(role) for role in scores.keys()],
                    "Score": [round(score, 2) for score in scores.values()],
                }
            )
            .sort_values("Score", ascending=False)
            .reset_index(drop=True)
        )

        st.dataframe(scores_df, use_container_width=True, hide_index=True)
        st.bar_chart(scores_df.set_index("Role"))

        st.markdown(
            """
            <p class="small-note">
            Note: This result is a recommendation based on the training data,
            not a final career decision.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as exc:
        st.error("Prediction failed. Please check that the saved model and vectorizer match.")
        st.exception(exc)

st.markdown("---")
st.caption("Academic project extension for Saudi Tech Job Skills Analysis.")

