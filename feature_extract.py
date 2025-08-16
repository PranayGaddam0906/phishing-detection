import re
import joblib
import numpy as np
import tldextract

__all__ = ["extract_features", "extract_all_features", "model"]

# ---------------------------------------------------
# Feature Extraction
# ---------------------------------------------------
def extract_features(url: str):
    features = {}

    # Basic cleanup
    domain_info = tldextract.extract(url)
    domain = domain_info.domain
    suffix = domain_info.suffix
    full_domain = domain_info.fqdn

    # 1. URL length
    features["LongURL"] = len(url) > 75
    features["ShortURL"] = len(url) < 20

    # 2. Using IP address (instead of domain)
    ip_pattern = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    features["UsingIP"] = bool(ip_pattern.match(domain))

    # 3. Numeric-only domain
    features["NumericOnly"] = domain.isdigit()

    # 4. Presence of '@' symbol
    features["Symbol@"] = "@" in url

    # 5. Redirecting with '//'
    features["Redirecting//"] = url.count("//") > 1

    # 6. Prefix/Suffix (-)
    features["PrefixSuffix-"] = "-" in domain

    # 7. Subdomains
    features["SubDomains"] = full_domain.count(".") > 2

    # 8. HTTPS
    features["HTTPS"] = url.lower().startswith("https")

    # You can expand further (favicon, ports, Google index, etc.)
    # Placeholder defaults for dataset compatibility
    features.update({
        'DomainRegLen': 1,
        'Favicon': 1,
        'NonStdPort': 0,
        'HTTPSDomainURL': 1,
        'RequestURL': 1,
        'AnchorURL': 1,
        'LinksInScriptTags': 1,
        'ServerFormHandler': 1,
        'InfoEmail': 0,
        'AbnormalURL': 0,
        'WebsiteForwarding': 0,
        'StatusBarCust': 0,
        'DisableRightClick': 0,
        'UsingPopupWindow': 0,
        'IframeRedirection': 0,
        'AgeofDomain': 1,
        'DNSRecording': 1,
        'WebsiteTraffic': 1,
        'PageRank': 1,
        'GoogleIndex': 1,
        'LinksPointingToPage': 0,
        'StatsReport': 0
    })

    return features

# ---------------------------------------------------
# Align features with training
# ---------------------------------------------------
def extract_all_features(url):
    features = extract_features(url)

    features_list = [
        'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
        'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
        'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
        'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
        'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',
        'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording',
        'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage',
        'StatsReport', 'NumericOnly'   # 🔹 new feature
    ]

    main_list = []
    for check in features_list:
        main_list.append(features.get(check, 0))  # default 0 if missing
    return main_list

# ---------------------------------------------------
# Model prediction
# ---------------------------------------------------
def model(feature_vector):
    loaded_model = joblib.load("hybrid_model.joblib")
    sample_input = np.array([feature_vector])  # shape (1, n_features)

    prediction = loaded_model.predict(sample_input)[0]
    probabilities = loaded_model.predict_proba(sample_input)[0]

    print(f"Predicted Class: {prediction}")
    print(f"Probabilities -> Legitimate: {probabilities[0]:.2f}, Phishing: {probabilities[1]:.2f}")

    return prediction, probabilities

# ---------------------------------------------------
# Test Run
# ---------------------------------------------------
if __name__ == "__main__":
    url = "9185780"
    features = extract_all_features(url)
    print("Extracted Features Length:", len(features))

    pred, proba = model(features)
    print("Final Prediction:", "Phishing" if pred == 1 else "Legitimate")
