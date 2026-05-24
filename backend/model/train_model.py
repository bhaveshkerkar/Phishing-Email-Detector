"""
Phishing Email Detection - Model Training
Ensemble: Random Forest + Gradient Boosting + Logistic Regression
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../../dataset/phishing_dataset.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "phishing_model.pkl")

# --- Features used for training ---
NUMERIC_FEATURES = [
    "subject_length", "text_length", "num_exclamations", "num_caps_words",
    "num_urgent_words", "has_attachment", "has_url", "url_shortener",
    "has_ip_url", "sender_domain_mismatch", "html_obfuscation",
    "num_links", "has_prize_words", "has_threat_words", "has_financial_words",
]


def load_data(path):
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Phishing : {df['label'].sum()}")
    print(f"   Legit    : {(df['label'] == 0).sum()}")
    df["combined_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    return df


def train(df):
    X_text = df["combined_text"]
    X_num  = df[NUMERIC_FEATURES].values
    y      = df["label"].values

    # Train/test split
    X_tr_t, X_te_t, X_tr_n, X_te_n, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF (unigrams + bigrams)
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = hstack([tfidf.fit_transform(X_tr_t), csr_matrix(X_tr_n)])
    X_test  = hstack([tfidf.transform(X_te_t),     csr_matrix(X_te_n)])

    # Ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ],
        voting="soft",
    )

    print("\n🔄 Training model...")
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    print("\n=== Results ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save model bundle
    bundle = {
        "model":            ensemble,
        "tfidf":            tfidf,
        "numeric_features": NUMERIC_FEATURES,
    }
    joblib.dump(bundle, MODEL_OUTPUT)
    print(f"\n✅ Model saved → {MODEL_OUTPUT}")


if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    train(df)