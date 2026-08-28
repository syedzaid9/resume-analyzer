"""Model Evaluation Script for ResumeAI.
Loads the serialized classifier and vectorizer, evaluates on held-out test data,
computes per-class classification reports, and extracts domain key features.
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    PROCESSED_DATASET_PATH,
    BEST_CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    MODEL_METADATA_PATH,
)


def evaluate_saved_model():
    """Evaluates the saved model and prints per-class performance."""
    if not BEST_CLASSIFIER_PATH.exists() or not TFIDF_VECTORIZER_PATH.exists():
        print("[ERROR] Model artifacts not found. Run scripts/train_model.py first.")
        sys.exit(1)

    print(f"[INFO] Loading model from: {BEST_CLASSIFIER_PATH}")
    model = joblib.load(BEST_CLASSIFIER_PATH)

    print(f"[INFO] Loading vectorizer from: {TFIDF_VECTORIZER_PATH}")
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    df = pd.read_csv(PROCESSED_DATASET_PATH)
    X = df["Cleaned_Resume"].astype(str).values
    y = df["Category"].astype(str).values

    # Reproduce same split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT ON TEST RESUMES")
    print("=" * 70)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")
    print(report)
    print("=" * 70)

    # Top feature analysis
    feature_names = np.array(vectorizer.get_feature_names_out())
    print("\nTOP INFORMATIVE TF-IDF KEYWORDS ACROSS VOCABULARY:")
    print(f"Total Vocabulary Size: {len(feature_names)} n-grams")
    sample_features = feature_names[:25]
    print(f"Sample Features: {', '.join(sample_features)}")

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"\n[INFO] Best Model Selected During Training: '{metadata.get('best_model_name')}'")


if __name__ == "__main__":
    evaluate_saved_model()
