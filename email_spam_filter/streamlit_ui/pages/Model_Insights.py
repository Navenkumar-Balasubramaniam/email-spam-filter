from __future__ import annotations

import streamlit as st

from email_spam_filter.inference import SpamClassifier
from email_spam_filter.viz import Visualizer

st.title("📊 Model Insights")

@st.cache_resource
def get_classifier():
    return SpamClassifier()

@st.cache_resource
def get_visualizer():
    return Visualizer()

clf = get_classifier()
viz = get_visualizer()

pipeline = clf.pipeline()

st.subheader("Top Spam Tokens")
fig_spam = viz.plot_top_tokens(pipeline, class_label="spam", n=15)
st.pyplot(fig_spam)

st.subheader("Top Ham Tokens")
fig_ham = viz.plot_top_tokens(pipeline, class_label="ham", n=15)
st.pyplot(fig_ham)

meta = clf.metadata
if meta:
    st.subheader("Model Metadata")
    st.json(meta)