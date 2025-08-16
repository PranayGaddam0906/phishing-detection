import os
import joblib
import streamlit as st
from feature_extract import extract_features
import train  # your train.py script

MODEL_FILE = "hybrid_model.joblib"

# Auto-train if model doesn't exist
st.title("🔒 Phishing Website Detection")
st.write("A machine learning app to detect phishing websites.")

# User input: website URL
url = st.text_input("Enter Website URL")

if st.button("Predict"):
    if url:
        try:
            # Extract features from the URL using your custom function
            features = extract_features(url)
            df = pd.DataFrame([features])

            # Predict
            prediction = model.predict(df)[0]

            if prediction == 1:
                st.error("🚨 Phishing Website Detected!")
            else:
                st.success("✅ Legitimate Website")
        except Exception as e:
            st.error(f"Error processing URL: {e}")
    else:
        st.warning("Please enter a URL first.")
