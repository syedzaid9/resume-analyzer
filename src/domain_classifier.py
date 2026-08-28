"""Domain Classifier Module for ResumeAI.
Loads the trained ML classification model and TF-IDF vectorizer to predict target professional domains
and calculate calibrated confidence scores for top matching domains.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import joblib
import numpy as np

from src.config import BEST_CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH
from src.text_preprocessor import preprocessor


class DomainClassifier:
    """Inferences the best matching job domain from resume text with confidence estimates."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model and vectorizer from disk."""
        if BEST_CLASSIFIER_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
            try:
                self.model = joblib.load(BEST_CLASSIFIER_PATH)
                self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            except Exception as e:
                print(f"[WARN] Failed to load ML model artifacts: {e}")
                self.model = None
                self.vectorizer = None

    def is_model_ready(self) -> bool:
        """Returns True if model and vectorizer are loaded and operational."""
        return self.model is not None and self.vectorizer is not None

    def predict_domain(self, raw_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Predicts the most probable domain and returns top-k ranking with genuine confidence scores."""
        if not self.is_model_ready():
            return {
                "success": False,
                "primary_domain": "Data Science (Default - Model Not Loaded)",
                "confidence_pct": 0.0,
                "top_domains": [],
                "error": "ML model artifacts not loaded.",
            }

        cleaned_text = preprocessor.clean_text_for_ml(raw_text)
        if not cleaned_text or len(cleaned_text.split()) < 5:
            return {
                "success": False,
                "primary_domain": "Insufficient Text",
                "confidence_pct": 0.0,
                "top_domains": [],
                "error": "Resume text is too brief to classify reliably.",
            }

        vec = self.vectorizer.transform([cleaned_text])

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(vec)[0]
            # Softmax calculation
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / np.sum(exp_d)
        else:
            # Fallback uniform
            classes = self.model.classes_
            probs = np.ones(len(classes)) / len(classes)

        classes = self.model.classes_
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_domains = []
        for idx in top_indices:
            domain_name = str(classes[idx])
            prob_pct = round(float(probs[idx]) * 100.0, 1)
            top_domains.append({
                "domain": domain_name,
                "confidence_pct": prob_pct,
            })

        primary_domain = top_domains[0]["domain"] if top_domains else "Unknown"
        confidence_pct = top_domains[0]["confidence_pct"] if top_domains else 0.0

        return {
            "success": True,
            "primary_domain": primary_domain,
            "confidence_pct": confidence_pct,
            "top_domains": top_domains,
            "error": None,
        }


# Global instance
domain_classifier = DomainClassifier()
