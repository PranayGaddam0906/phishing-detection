import streamlit as st
import numpy as np
from feature_extract import extract_all_features,model



st.title("Phishing URL Detection")

url = st.text_input("Enter the URL")

if st.button("Detect"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        try:
            feature = extract_all_features(url)
            try:
                result = model(feature)  # Pass model_path
                if result==-1:
                    st.markdown(f"<span style='color:red;'>**Result:** Phishing</span>", unsafe_allow_html=True)
                elif result==1:
                    st.markdown(f"<span style='color:green;'>**Result:** Not Phishing</span>", unsafe_allow_html=True)
                else:  # Handle errors or unexpected results
                    st.write(f"**Result:** {result}")  # Display the error message

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

        except Exception as e:
            st.error(f"An error occurred during feature extraction: {e}")