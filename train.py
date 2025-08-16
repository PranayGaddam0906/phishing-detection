import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from feature_extract import extract_features   # ✅ central feature extractor

MODEL_FILE = "hybrid_model.joblib"

def train_model():
    # Load dataset
    data = pd.read_csv("phishing.csv")

    # ✅ Extract features from every URL in dataset
    if "url" in data.columns:
        feature_df = data["url"].apply(extract_features).apply(pd.Series)
        data = pd.concat([data, feature_df], axis=1)
    else:
        raise ValueError("❌ Dataset does not have 'url' column")

    # Features & Labels
    X = data.drop(["Index", "class", "url"], axis=1, errors="ignore")
    y = data["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Hybrid: Soft Voting
    hybrid = VotingClassifier(
        estimators=[("knn", knn_pipeline), ("xgb", xgb_model)],
        voting="soft"
    )

    # Train
    hybrid.fit(X_train, y_train)

    # Evaluate
    y_pred = hybrid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Hybrid Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(hybrid, MODEL_FILE)
    print(f"💾 Hybrid model saved as {MODEL_FILE}")

    return hybrid

if __name__ == "__main__":
    train_model()
