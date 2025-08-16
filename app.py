import os
import joblib
import streamlit as st
import pandas as pd

from feature_extract import extract_all_features
import train  # <-- your train.py with train_model() defined

MODEL_FILE = "hybrid_model.joblib"

# Title
st.title("🔒 Phishing Website Detection")
st.write("A machine learning app to detect phishing websites.")

# 🔹 Load or train model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.warning("Model not found. Training a new one...")
    model = train.train_model()  # <-- now this works because we fixed train.py
    joblib.dump(model, MODEL_FILE)

# User input
url = st.text_input("Enter Website URL")

if st.button("Predict"):
    if url:
        try:
            # Extract features from the URL
            features = extract_all_features(url)
            df = pd.DataFrame([features])
            prediction = model.predict([features])[0] 

            if prediction == 1:
                st.error("🚨 Phishing Website Detected!")
            else:
                st.success("✅ Legitimate Website")
        except Exception as e:
            st.error(f"Error processing URL: {e}")
    else:
        st.warning("Please enter a URL first.")
