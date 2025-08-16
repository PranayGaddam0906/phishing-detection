import os
import joblib
import streamlit as st
from feature_extract import extract_features
import train  # your train.py script

MODEL_FILE = "hybrid_model.joblib"

# Auto-train if model doesn't exist
if not os.path.exists(MODEL_FILE):
    st.warning("⚠️ Model not found — training a new one, please wait...")
    train.main()  # assuming train.py has a main() function
model = joblib.load(MODEL_FILE)

st.title("🔒 Phishing Website Detection")

