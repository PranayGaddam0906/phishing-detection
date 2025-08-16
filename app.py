import streamlit as st
import joblib
import pandas as pd
from feature_extract import get_features
features = get_features(user_url)


# Load trained model
model = joblib.load("hybrid_model.joblib")


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
