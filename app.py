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
    # Social Media
    "linkedin.com", "facebook.com", "instagram.com", "twitter.com", "reddit.com",
    "tiktok.com", "snapchat.com", "pinterest.com", "whatsapp.com", "telegram.org",
    
    # Tech / Developer
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com", "medium.com",
    "quora.com", "wikipedia.org", "leetcode.com", "hackerrank.com", "geeksforgeeks.org",
    "kaggle.com", "codeforces.com", "topcoder.com", "codechef.com", "dev.to",
    
    # Search / Email
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com", "outlook.com", "protonmail.com",
    
    # E-commerce / Marketplaces
    "amazon.com", "flipkart.com", "ebay.com", "aliexpress.com", "etsy.com", "walmart.com",
    "bestbuy.com", "target.com", "shopify.com", "craigslist.org",
    
    # News / Media
    "bbc.com", "cnn.com", "nytimes.com", "theguardian.com", "forbes.com",
    "bloomberg.com", "reuters.com", "huffpost.com", "washingtonpost.com", "buzzfeed.com",
    
    # Entertainment / Streaming
    "netflix.com", "youtube.com", "spotify.com", "disneyplus.com", "hulu.com",
    "primevideo.com", "twitch.tv", "soundcloud.com", "vimeo.com", "imdb.com",
    
    # Education
    "coursera.org", "edx.org", "udemy.com", "khanacademy.org", "udacity.com",
    "futurelearn.com", "skillshare.com", "academic.microsoft.com", "ocw.mit.edu", "openlearning.com",
    
    # Banking / Finance
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com", "hsbc.com",
    "citibank.com", "americanexpress.com", "discover.com", "capitalone.com", "stripe.com",
    
    # Misc / Popular Services
    "apple.com", "microsoft.com", "adobe.com", "dropbox.com", "slack.com",
    "zoom.us", "canva.com", "notion.so", "figma.com", "asana.com",
    
    # Travel / Booking
    "booking.com", "airbnb.com", "expedia.com", "tripadvisor.com", "uber.com",
    "lyft.com", "hotels.com", "kayak.com", "skyscanner.com", "trivago.com"
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
