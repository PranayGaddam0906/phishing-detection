import streamlit as st
import joblib
import pandas as pd
from feature_extract import extract_features

# Load model
model = joblib.load("hybrid_model.joblib")

st.title("🔎 Phishing Website Detection App")

# Input box
user_url = st.text_input("Enter a URL to check:")

if st.button("Check URL"):
    if user_url:
        # Extract features for the given URL
        features = extract_features(user_url)

        # Convert features into DataFrame for model
        X = pd.DataFrame([features])

        # Predict
        prediction = model.predict(X)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This looks like a **legitimate website**.")
        else:
            st.error("⚠️ Warning: This might be a **phishing website**!")
    else:
        st.warning("Please enter a URL first.")
