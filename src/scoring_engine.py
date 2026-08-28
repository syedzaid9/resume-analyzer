"""Transparent Resume Scoring Engine for ResumeAI.
Computes weighted component scores (Skill Match, Experience, Projects, Education, ATS, Structure, Formatting)
using strictly configurable weights from src/config.py.
Outputs total 'AI Resume Fit Score' and comprehensive rating breakdown.
"""

from typing import Dict, List, Any
import pandas as pd

from src.config import (
    SCORING_WEIGHTS,
    RATING_TIERS,
    DOMAIN_MAPPING_CSV_PATH,
    IMPORTANCE_LEVELS,
)


class ScoringEngine:
    """Calculates granular component scores and overall AI Resume Fit Score."""

    def __init__(self):
        self.domain_skills_df = None
        self._load_domain_skills()

    def _load_domain_skills(self):
        """Loads domain skill mappings and importance levels."""
        if DOMAIN_MAPPING_CSV_PATH.exists():
            self.domain_skills_df = pd.read_csv(DOMAIN_MAPPING_CSV_PATH)

    def calculate_skill_alignment(
        self, detected_skills: List[str], target_domain: str
    ) -> Dict[str, Any]:
        """Calculates matched vs missing domain skills weighted by importance level."""
        if self.domain_skills_df is None or target_domain not in self.domain_skills_df["domain_name"].values:
            # Fallback if domain mapping is not available
            total_detected = len(detected_skills)
            pct = min(total_detected * 10.0, 100.0)
            return {
                "skill_match_pct": pct,
                "matched_skills": detected_skills[:8],
                "missing_skills": [],
                "matched_count": len(detected_skills),
                "total_target_skills": 10,
                "missing_by_importance": {},
            }

        domain_rows = self.domain_skills_df[self.domain_skills_df["domain_name"] == target_domain]
        detected_set = {s.lower() for s in detected_skills}

        matched_skills: List[Dict[str, Any]] = []
        missing_skills: List[Dict[str, Any]] = []
        missing_by_importance: Dict[str, List[str]] = {imp: [] for imp in IMPORTANCE_LEVELS}

        total_possible_weight = 0.0
        earned_weight = 0.0

        for _, row in domain_rows.iterrows():
            skill_name = str(row["skill_name"]).strip()
            importance = str(row.get("importance", "Medium")).strip()
            category = str(row.get("category", "General")).strip()

            imp_info = IMPORTANCE_LEVELS.get(importance, IMPORTANCE_LEVELS["Medium"])
            w = imp_info["weight"]
            total_possible_weight += w

            if skill_name.lower() in detected_set:
                earned_weight += w
                matched_skills.append({
                    "skill_name": skill_name,
                    "importance": importance,
                    "category": category,
                })
            else:
                missing_skills.append({
                    "skill_name": skill_name,
                    "importance": importance,
                    "category": category,
                })
                missing_by_importance[importance].append(skill_name)

        if total_possible_weight > 0:
            match_pct = round((earned_weight / total_possible_weight) * 100.0, 1)
        else:
            match_pct = 50.0

        # Sort missing skills by priority (Critical -> High -> Medium -> Low)
        missing_skills.sort(
            key=lambda x: IMPORTANCE_LEVELS.get(x["importance"], {}).get("priority", 99)
        )

        return {
            "skill_match_pct": match_pct,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_count": len(matched_skills),
            "total_target_skills": len(domain_rows),
            "missing_by_importance": missing_by_importance,
        }

    def compute_overall_score(
        self,
        resume_text: str,
        parsed_data: Dict[str, Any],
        detected_skills: List[str],
        ats_data: Dict[str, Any],
        target_domain: str,
    ) -> Dict[str, Any]:
        """Calculates 7-component transparent scoring breakdown and overall score."""
        skill_align = self.calculate_skill_alignment(detected_skills, target_domain)
        skill_match_pct = skill_align["skill_match_pct"]

        personal_info = parsed_data.get("personal_info", {})
        education_info = parsed_data.get("education", {})
        experience_info = parsed_data.get("experience", {})
        projects_info = parsed_data.get("projects", {})
        sections = parsed_data.get("sections", {})

        # 1. Skill Match Component (30%)
        skill_score_val = (skill_match_pct / 100.0) * SCORING_WEIGHTS["skill_match"]

        # 2. Experience Component (20%)
        exp_pct = 30.0  # Base
        if experience_info.get("has_experience_section"):
            exp_pct += 25.0
        if experience_info.get("detected_titles") and experience_info["detected_titles"][0] != "Not detected":
            exp_pct += 20.0
        if experience_info.get("action_verbs_count", 0) >= 3:
            exp_pct += 15.0
        if experience_info.get("metrics_count", 0) >= 1:
            exp_pct += 10.0
        exp_pct = min(exp_pct, 100.0)
        exp_score_val = (exp_pct / 100.0) * SCORING_WEIGHTS["experience"]

        # 3. Projects Component (15%)
        proj_pct = 20.0
        if projects_info.get("has_projects_section"):
            proj_pct += 35.0
        if projects_info.get("project_lines_count", 0) >= 3:
            proj_pct += 25.0
        if projects_info.get("project_metrics_detected"):
            proj_pct += 20.0
        proj_pct = min(proj_pct, 100.0)
        proj_score_val = (proj_pct / 100.0) * SCORING_WEIGHTS["projects"]

        # 4. Education Component (10%)
        edu_pct = 40.0
        if education_info.get("has_degree"):
            edu_pct += 30.0
        if education_info.get("majors") and education_info["majors"][0] != "Not detected":
            edu_pct += 20.0
        if education_info.get("grad_years") and education_info["grad_years"][0] != "Not detected":
            edu_pct += 10.0
        edu_pct = min(edu_pct, 100.0)
        edu_score_val = (edu_pct / 100.0) * SCORING_WEIGHTS["education"]

        # 5. ATS Compatibility Component (10%)
        ats_score_val = (ats_data.get("ats_score", 70) / 100.0) * SCORING_WEIGHTS["ats_compatibility"]

        # 6. Resume Structure Component (10%)
        struct_count = sum(1 for k in ["summary", "skills", "experience", "education", "projects"] if sections.get(k))
        struct_pct = min((struct_count / 4.0) * 100.0, 100.0)
        struct_score_val = (struct_pct / 100.0) * SCORING_WEIGHTS["structure"]

        # 7. Formatting & Readability Component (5%)
        words = len(resume_text.split())
        fmt_pct = 100.0 if 300 <= words <= 1600 else (70.0 if words >= 150 else 40.0)
        fmt_score_val = (fmt_pct / 100.0) * SCORING_WEIGHTS["formatting"]

        # Sum overall score
        total_score = round(
            skill_score_val
            + exp_score_val
            + proj_score_val
            + edu_score_val
            + ats_score_val
            + struct_score_val
            + fmt_score_val,
            1,
        )
        total_score_int = int(round(total_score))
        total_score_int = min(max(total_score_int, 0), 100)

        # Rating tier lookup
        rating_tier = next(
            (t for t in RATING_TIERS if t["min"] <= total_score_int <= t["max"]),
            RATING_TIERS[-1],
        )

        breakdown = {
            "skill_match": {"earned": round(skill_score_val, 1), "max": SCORING_WEIGHTS["skill_match"], "pct": skill_match_pct},
            "experience": {"earned": round(exp_score_val, 1), "max": SCORING_WEIGHTS["experience"], "pct": exp_pct},
            "projects": {"earned": round(proj_score_val, 1), "max": SCORING_WEIGHTS["projects"], "pct": proj_pct},
            "education": {"earned": round(edu_score_val, 1), "max": SCORING_WEIGHTS["education"], "pct": edu_pct},
            "ats_compatibility": {"earned": round(ats_score_val, 1), "max": SCORING_WEIGHTS["ats_compatibility"], "pct": ats_data.get("ats_score", 70)},
            "structure": {"earned": round(struct_score_val, 1), "max": SCORING_WEIGHTS["structure"], "pct": struct_pct},
            "formatting": {"earned": round(fmt_score_val, 1), "max": SCORING_WEIGHTS["formatting"], "pct": fmt_pct},
        }

        return {
            "overall_fit_score": total_score_int,
            "rating_label": rating_tier["label"],
            "rating_color": rating_tier["color"],
            "rating_badge": rating_tier["badge"],
            "breakdown": breakdown,
            "skill_alignment": skill_align,
            "weights_used": SCORING_WEIGHTS,
        }


# Global instance
scoring_engine = ScoringEngine()
