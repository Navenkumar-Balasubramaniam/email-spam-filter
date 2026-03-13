from __future__ import annotations

import streamlit as st

from email_spam_filter.inference import SpamClassifier
from email_spam_filter.viz import Visualizer

st.set_page_config(
    page_title="Email Spam Filter",
    layout="wide",
)

st.title("📧 Email Spam Filter")
st.caption("Pretrained spam classifier with interactive insights")

# Initialize once
@st.cache_resource
def get_classifier():
    return SpamClassifier()

@st.cache_resource
def get_visualizer():
    return Visualizer()

clf = get_classifier()
viz = get_visualizer()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Spam Threshold", 0.0, 1.0, 0.5, 0.01)

st.markdown("### Single Email Prediction")

text_input = st.text_area("Enter email text:", height=200)

if st.button("Predict", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = clf.classify(text_input, threshold=threshold)
        label = result.label.upper()

        if label == "SPAM":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric("Spam Probability", f"{result.spam_probability:.4f}")
        st.progress(min(max(result.spam_probability, 0.0), 1.0))

st.divider()
st.markdown("Use the sidebar pages for batch prediction and model insights.")