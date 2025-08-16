import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

MODEL_FILE = "hybrid_model.joblib"


# 🔹 Custom transformer to add NumericOnly feature
class AddNumericOnly(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if "url" in X.columns:
            X["NumericOnly"] = X["url"].apply(
                lambda u: int(str(u).replace("http://", "")
                                       .replace("https://", "")
                                       .replace("www.", "")
                                       .split("/")[0]
                                       .isdigit())
            )
            X = X.drop(columns=["url"])
        else:
            if "NumericOnly" not in X.columns:
                X["NumericOnly"] = 0
        return X


def train_model():
    # Load dataset
    data = pd.read_csv("phishing.csv")

    # Features & Labels
    X = data.drop(["Index", "class"], axis=1, errors="ignore")
    y = data["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define base models
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
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # 🔹 Wrap everything in a single pipeline
    hybrid = Pipeline([
        ("add_numeric", AddNumericOnly()),
        ("voting", VotingClassifier(
            estimators=[("knn", knn_pipeline), ("xgb", xgb_model)],
            voting="soft"
        ))
    ])

    # Train
    hybrid.fit(X_train, y_train)

    # Evaluate
    y_pred = hybrid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Hybrid Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the whole pipeline
    joblib.dump(hybrid, MODEL_FILE)
    print(f"💾 Hybrid model saved as {MODEL_FILE}")

    return hybrid


if __name__ == "__main__":
    train_model()
