"""Streamlit app for CFPB complaint classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_logistic_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="CFPB Complaint Classifier", page_icon="📊", layout="wide")

st.title("CFPB Consumer Complaint Classifier")
st.write(
    "This app uses Natural Language Processing to classify a consumer complaint "
    "narrative into a CFPB financial product category."
)

with st.sidebar:
    st.header("Project Info")
    st.markdown("""
    **Model:** TF-IDF + Logistic Regression  
    **Dataset:** CFPB Consumer Complaint Database  
    **Use Case:** Complaint routing and financial customer experience analytics
    """)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run `python src/train_model.py` first to create "
        "models/tfidf_logistic_model.joblib."
    )
    st.stop()

model = load_model()

def predict_with_confidence(text: str):
    prediction = model.predict([text])[0]
    confidence_df = None
    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        labels = model.named_steps["model"].classes_
        confidence_df = pd.DataFrame({"Product": labels, "Confidence": probabilities})
        confidence_df = confidence_df.sort_values("Confidence", ascending=False).head(5)
    return prediction, confidence_df

complaint_text = st.text_area(
    "Paste a complaint narrative",
    height=220,
    placeholder="Example: I disputed inaccurate information on my credit report but the bureau continues to verify accounts that do not belong to me...",
)

if st.button("Classify Complaint", type="primary"):
    if not complaint_text.strip():
        st.warning("Please enter a complaint narrative first.")
    else:
        prediction, confidence_df = predict_with_confidence(complaint_text)
        st.subheader("Predicted Product Category")
        st.success(prediction)

        if confidence_df is not None:
            st.subheader("Top Confidence Scores")
            st.dataframe(confidence_df, use_container_width=True)
            st.bar_chart(confidence_df.set_index("Product"))
