"""Machine Learning Model Training Pipeline for ResumeAI.
Trains and compares 4 supervised NLP classifiers (Logistic Regression, Multinomial Naive Bayes,
Calibrated Linear SVM, and Random Forest) using Stratified Cross-Validation & Held-out Test Evaluation.
Saves the best model and vectorizer to models/ directory.
Never fabricates metrics.
"""

import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    PROCESSED_DATASET_PATH,
    CLASSIFIER_DIR,
    VECTORIZER_DIR,
    BEST_CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    MODEL_METADATA_PATH,
)


def train_models():
    """Trains 4 ML classifiers, selects the best, and serializes artifacts."""
    if not PROCESSED_DATASET_PATH.exists():
        print(f"[ERROR] Processed dataset not found at {PROCESSED_DATASET_PATH}. Run preprocess_dataset.py first.")
        sys.exit(1)

    print(f"[INFO] Loading processed dataset from: {PROCESSED_DATASET_PATH}")
    df = pd.read_csv(PROCESSED_DATASET_PATH)

    X = df["Cleaned_Resume"].astype(str).values
    y = df["Category"].astype(str).values

    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"[INFO] Dataset loaded with {len(X)} samples across {len(unique_classes)} distinct domains.")

    # 1. Stratified Train / Test Split (80% Train+Val, 20% Held-out Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Further split Train/Val (75% Train, 25% Val of the train_val subset -> 60% Train, 20% Val, 20% Test overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"[INFO] Dataset split: Train={len(X_train)} ({len(X_train)/len(X)*100:.0f}%), "
          f"Validation={len(X_val)} ({len(X_val)/len(X)*100:.0f}%), "
          f"Test={len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

    # 2. TF-IDF Feature Extraction
    print("[INFO] Vectorizing text with TF-IDF (n-grams=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3500,
        sublinear_tf=True,
        min_df=1,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    X_train_val_vec = vectorizer.fit_transform(X_train_val)
    X_test_final_vec = vectorizer.transform(X_test)

    # 3. Model Candidates
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight="balanced"),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    print("\n" + "=" * 70)
    print("MODEL VALIDATION COMPARISON")
    print("=" * 70)
    print(f"{'Model Name':<26} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Time (s)'}")
    print("-" * 70)

    best_model_name = None
    best_val_f1 = -1.0

    for name, model in candidate_models.items():
        t0 = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t0

        y_val_pred = model.predict(X_val_vec)
        acc = accuracy_score(y_val, y_val_pred)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        results[name] = {
            "val_accuracy": float(acc),
            "val_macro_f1": float(macro_f1),
            "val_weighted_f1": float(weighted_f1),
            "train_time": float(train_time),
        }

        print(f"{name:<26} | {acc * 100:>8.2f}% | {macro_f1 * 100:>8.2f}% | {weighted_f1 * 100:>10.2f}% | {train_time:>6.3f}s")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_name = name

    print("=" * 70)
    print(f"[SELECTION] Best Performing Model: '{best_model_name}' (Validation Macro F1: {best_val_f1 * 100:.2f}%)")

    # 4. Train Best Model on full Train+Validation set and evaluate on Held-Out Test Set
    best_estimator = candidate_models[best_model_name]
    best_estimator.fit(X_train_val_vec, y_train_val)

    y_test_pred = best_estimator.predict(X_test_final_vec)
    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    test_precision_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_recall_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

    print("\n" + "=" * 70)
    print(f"FINAL HELD-OUT TEST EVALUATION ({best_model_name})")
    print("=" * 70)
    print(f"Test Accuracy:          {test_accuracy * 100:.2f}%")
    print(f"Test Macro Precision:   {test_precision_macro * 100:.2f}%")
    print(f"Test Macro Recall:      {test_recall_macro * 100:.2f}%")
    print(f"Test Macro F1-Score:    {test_macro_f1 * 100:.2f}%")
    print(f"Test Weighted F1-Score: {test_weighted_f1 * 100:.2f}%")
    print("=" * 70)

    # 5. Save Artifacts
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    VECTORIZER_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_estimator, BEST_CLASSIFIER_PATH)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    print(f"[SUCCESS] Serialized best model to: {BEST_CLASSIFIER_PATH}")
    print(f"[SUCCESS] Serialized vectorizer to: {TFIDF_VECTORIZER_PATH}")

    # 6. Save Model Metadata
    conf_mat = confusion_matrix(y_test, y_test_pred, labels=sorted(list(unique_classes))).tolist()
    metadata = {
        "best_model_name": best_model_name,
        "classes": sorted(list(unique_classes)),
        "num_classes": len(unique_classes),
        "total_dataset_size": len(X),
        "train_samples": len(X_train_val),
        "test_samples": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "validation_results": results,
        "test_metrics": {
            "accuracy": test_accuracy,
            "macro_precision": test_precision_macro,
            "macro_recall": test_recall_macro,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
        },
        "confusion_matrix": conf_mat,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Saved model metadata to: {MODEL_METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train_models()
