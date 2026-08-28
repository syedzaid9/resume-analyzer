"""Semantic Matcher Module for ResumeAI.
Computes contextual and TF-IDF cosine similarity between a candidate resume representation
and the target domain's occupational profile.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import OCCUPATIONS_CSV_PATH, DOMAIN_MAPPING_CSV_PATH
from src.text_preprocessor import preprocessor


class SemanticMatcher:
    """Calculates semantic similarity between resume content and domain occupational profiles."""

    def __init__(self):
        self.domain_profiles: Dict[str, str] = {}
        self._build_domain_profiles()

    def _build_domain_profiles(self):
        """Constructs rich contextual text representations for each domain."""
        if not OCCUPATIONS_CSV_PATH.exists():
            return

        df_occ = pd.read_csv(OCCUPATIONS_CSV_PATH)
        df_map = pd.read_csv(DOMAIN_MAPPING_CSV_PATH) if DOMAIN_MAPPING_CSV_PATH.exists() else None

        for _, row in df_occ.iterrows():
            domain = str(row["domain_name"]).strip()
            desc = str(row.get("description", ""))
            edu = str(row.get("education_expectation", ""))
            exp = str(row.get("experience_expectation", ""))
            roles = str(row.get("related_roles", ""))

            skills_text = ""
            if df_map is not None:
                domain_skills = df_map[df_map["domain_name"] == domain]["skill_name"].tolist()
                skills_text = " ".join(domain_skills)

            combined_profile = f"{domain} {desc} {edu} {exp} {roles} {skills_text}"
            self.domain_profiles[domain] = preprocessor.clean_text_for_ml(combined_profile)

    def calculate_domain_similarity(self, resume_text: str, target_domain: str) -> Dict[str, Any]:
        """Calculates cosine similarity between resume and target domain profile."""
        domain_profile = self.domain_profiles.get(target_domain, "")

        if not domain_profile:
            # Fallback if domain is not found in database
            domain_profile = target_domain

        cleaned_resume = preprocessor.clean_text_for_ml(resume_text)

        if not cleaned_resume or len(cleaned_resume.split()) < 5:
            return {
                "similarity_score_pct": 0.0,
                "domain": target_domain,
                "disclaimer": "Semantic similarity is an analytical signal comparing vocabulary and contextual alignment with the target role, not a guarantee of employment.",
            }

        # Vectorize both
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            tfidf_matrix = vec.fit_transform([cleaned_resume, domain_profile])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            sim_pct = float(np.clip(sim * 100.0 * 1.35, 0.0, 100.0))  # Scale appropriately for natural resume variance
            sim_pct = round(sim_pct, 1)
        except Exception:
            sim_pct = 50.0

        return {
            "similarity_score_pct": sim_pct,
            "domain": target_domain,
            "disclaimer": "Semantic similarity is an analytical signal comparing vocabulary and contextual alignment with the target role, not a guarantee of employment.",
        }


# Global instance
semantic_matcher = SemanticMatcher()
