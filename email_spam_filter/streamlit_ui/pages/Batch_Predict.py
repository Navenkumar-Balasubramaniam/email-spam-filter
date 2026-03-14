from __future__ import annotations

import pandas as pd
import streamlit as st

from email_spam_filter.inference import SpamClassifier
from email_spam_filter.viz import Visualizer

st.title("📂 Batch Prediction")

@st.cache_resource
def get_classifier():
    return SpamClassifier()

@st.cache_resource
def get_visualizer():
    return Visualizer()

clf = get_classifier()
viz = get_visualizer()

uploaded_file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        texts = df["text"].astype(str).tolist()

        preds = clf.predict(texts)
        probs = clf.spam_probability(texts)

        df["prediction"] = preds
        df["spam_probability"] = probs

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Class Distribution")
        fig_counts = viz.plot_class_counts(df["prediction"].tolist())
        st.pyplot(fig_counts)

        st.subheader("Spam Probability Distribution")
        fig_hist = viz.plot_probability_histogram(df["spam_probability"].tolist())
        st.pyplot(fig_hist)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            csv,
            "classified_emails.csv",
            "text/csv"
        )