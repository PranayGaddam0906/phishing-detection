import os
import joblib
import streamlit as st
import pandas as pd
from urllib.parse import urlparse

from feature_extract import extract_all_features
import train  # your train.py with train_model() defined

MODEL_FILE = "hybrid_model.joblib"

# Title
st.title("🔒 Phishing Website Detection")
st.write("A machine learning app to detect phishing websites.")

# Extended safe domain whitelist
SAFE_DOMAINS = [
    "linkedin.com", "github.com", "google.com", "twitter.com",
    "facebook.com", "instagram.com", "youtube.com", "reddit.com",
    "stackoverflow.com", "medium.com", "quora.com", "wikipedia.org",
    "amazon.com", "apple.com", "microsoft.com", "netflix.com"
]

# 🔹 Load or train model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.warning("Model not found. Training a new one...")
    model = train.train_model()
    joblib.dump(model, MODEL_FILE)

# User input
url = st.text_input("Enter Website URL")

if st.button("Predict"):
    if url:
        try:
            # Extract domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Check whitelist first
            if any(sd in domain for sd in SAFE_DOMAINS):
                st.success("✅ Legitimate Website (Whitelisted)")
            else:
                # Extract features
                features = extract_all_features(url)
                df = pd.DataFrame([features])
                
                # Show features for inspection
                st.write("🔹 Feature Vector:", features)

                # Predict
                prediction = model.predict([features])[0]

                if prediction == 1:
                    st.error("🚨 Phishing Website Detected!")
                else:
                    st.success("✅ Legitimate Website")
        except Exception as e:
            st.error(f"Error processing URL: {e}")
    else:
        st.warning("Please enter a URL first.")
