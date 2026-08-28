"""Job Description Matcher Module for ResumeAI.
Compares candidate resume text against an arbitrary pasted Job Description (JD),
extracts JD skills, calculates match percentages, identifies missing keywords, and produces tailored suggestions.
"""

from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.skill_extractor import skill_extractor
from src.text_preprocessor import preprocessor


class JobMatcher:
    """Matches candidate resume against specific Job Description requirements."""

    def __init__(self):
        pass

    def match_job_description(
        self, resume_text: str, jd_text: str, resume_skills: List[str]
    ) -> Dict[str, Any]:
        """Performs deep comparison between resume and job description."""
        if not jd_text or len(jd_text.strip().split()) < 10:
            return {
                "success": False,
                "error": "Job description is too brief. Please paste a full job description.",
                "match_pct": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "recommendations": [],
            }

        # 1. Extract skills from Job Description
        jd_skills_data = skill_extractor.extract_skills(jd_text)
        jd_skills = jd_skills_data["all_detected_skills"]

        resume_skills_set = {s.lower() for s in resume_skills}
        matched_skills = []
        missing_skills = []

        for skill in jd_skills:
            if skill.lower() in resume_skills_set:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

        # 2. Compute Semantic Cosine Similarity
        cleaned_resume = preprocessor.clean_text_for_ml(resume_text)
        cleaned_jd = preprocessor.clean_text_for_ml(jd_text)

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            tfidf_mat = vec.fit_transform([cleaned_resume, cleaned_jd])
            cos_sim = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
            semantic_score = float(np.clip(cos_sim * 100.0 * 1.3, 0.0, 100.0))
        except Exception:
            semantic_score = 50.0

        # 3. Compute Skill Overlap Score
        if jd_skills:
            skill_score = (len(matched_skills) / len(jd_skills)) * 100.0
        else:
            skill_score = semantic_score

        # Weighted combined Job Match Score (60% Skill Match, 40% Semantic Content Match)
        final_match_pct = round(0.60 * skill_score + 0.40 * semantic_score, 1)
        final_match_pct = min(max(final_match_pct, 5.0), 99.0)

        # 4. Generate Tailored Recommendations
        recommendations = []
        if missing_skills:
            recommendations.append(
                f"The job description explicitly mentions: {', '.join(missing_skills[:4])}. "
                "Highlight these technologies in your skills and project descriptions if you possess verified experience."
            )
        if matched_skills:
            recommendations.append(
                f"Emphasize your proven background in {', '.join(matched_skills[:3])} in your summary and most recent experience bullets."
            )
        recommendations.append(
            "Mirror relevant industry keywords from the job description naturally throughout your resume bullets."
        )

        return {
            "success": True,
            "match_pct": final_match_pct,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "total_jd_skills": len(jd_skills),
            "recommendations": recommendations,
            "error": None,
        }


# Global instance
job_matcher = JobMatcher()
