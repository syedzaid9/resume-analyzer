"""ATS Analyzer Module for ResumeAI.
Audits resume content against Applicant Tracking System (ATS) parsing rules,
evaluates section headings, contact completeness, keyword density, and formatting.
"""

from typing import Dict, List, Any
from src.config import POWER_ACTION_VERBS


class ATSAnalyzer:
    """Evaluates ATS parsing compatibility and structural quality."""

    def __init__(self):
        pass

    def analyze(
        self,
        resume_text: str,
        parsed_data: Dict[str, Any],
        detected_skills: List[str],
        target_domain: str = "",
    ) -> Dict[str, Any]:
        """Performs a comprehensive ATS audit and returns an ATS score out of 100 with actionable feedback."""
        score = 0
        max_score = 100

        strengths: List[str] = []
        issues: List[str] = []
        suggestions: List[str] = []

        personal_info = parsed_data.get("personal_info", {})
        education_info = parsed_data.get("education", {})
        experience_info = parsed_data.get("experience", {})
        projects_info = parsed_data.get("projects", {})
        sections = parsed_data.get("sections", {})

        # 1. Contact Information Check (Max 25 pts)
        contact_pts = 0
        if personal_info.get("email") != "Not detected":
            contact_pts += 10
        else:
            issues.append("No professional email address detected.")
            suggestions.append("Add a clear email address at the top of your resume (e.g., name@domain.com).")

        if personal_info.get("phone") != "Not detected":
            contact_pts += 10
        else:
            issues.append("No phone number detected.")
            suggestions.append("Include an active phone number with country/area code.")

        if personal_info.get("linkedin") != "Not detected" or personal_info.get("github") != "Not detected":
            contact_pts += 5
            strengths.append("Professional profile links (LinkedIn/GitHub) detected.")
        else:
            issues.append("Missing professional online profiles (LinkedIn, GitHub, or Portfolio).")
            suggestions.append("Add your LinkedIn URL and GitHub/Portfolio to showcase verified work.")

        if contact_pts >= 20:
            strengths.append("Complete contact information header detected.")
        score += contact_pts

        # 2. Standard Section Headings Check (Max 30 pts)
        structure_pts = 0
        detected_headings = []

        if sections.get("skills") or len(detected_skills) >= 3:
            structure_pts += 10
            detected_headings.append("Skills")
        else:
            issues.append("Dedicated Skills section not clearly identified.")
            suggestions.append("Use a standard 'Technical Skills' heading for ATS keyword parsers.")

        if sections.get("experience") or experience_info.get("has_experience_section"):
            structure_pts += 10
            detected_headings.append("Experience")
        else:
            issues.append("Work Experience section heading is absent or non-standard.")
            suggestions.append("Use standard heading 'Work Experience' or 'Professional Experience'.")

        if sections.get("education") or education_info.get("has_degree"):
            structure_pts += 10
            detected_headings.append("Education")
        else:
            issues.append("Education credentials not clearly detected.")
            suggestions.append("Include an 'Education' section listing degree, institution, and graduation year.")

        if structure_pts == 30:
            strengths.append(f"Standard ATS headings present ({', '.join(detected_headings)}).")
        score += structure_pts

        # 3. Technical Skills & Keyword Coverage (Max 20 pts)
        skill_count = len(detected_skills)
        if skill_count >= 10:
            score += 20
            strengths.append(f"Strong keyword coverage ({skill_count} relevant technical skills extracted).")
        elif skill_count >= 5:
            score += 14
            strengths.append(f"Adequate keyword coverage ({skill_count} technical skills identified).")
        else:
            score += 6
            issues.append(f"Low keyword density (only {skill_count} recognized skills detected).")
            suggestions.append("Expand your skills section with relevant tools, frameworks, and programming languages.")

        # 4. Action Verbs & Experience Impact (Max 15 pts)
        action_verb_count = experience_info.get("action_verbs_count", 0)
        metrics_count = experience_info.get("metrics_count", 0)

        if action_verb_count >= 5:
            score += 10
            strengths.append(f"Strong action-verb usage ({action_verb_count} dynamic power verbs identified).")
        elif action_verb_count >= 2:
            score += 6
        else:
            issues.append("Weak action-oriented phrasing in work experience bullets.")
            suggestions.append("Begin accomplishment bullets with strong action verbs (e.g., 'Engineered', 'Automated', 'Scaled').")

        if metrics_count >= 2:
            score += 5
            strengths.append(f"Quantified achievements detected ({metrics_count} measurable metrics/percentages found).")
        else:
            issues.append("Experience bullets lack measurable business metrics or percentage improvements.")
            suggestions.append("Add measurable outcomes (e.g., 'increased throughput by 25%', 'reduced latency by 40ms') if verified.")

        # 5. Length & Readability (Max 10 pts)
        word_count = len(resume_text.split())
        if 250 <= word_count <= 1800:
            score += 10
            strengths.append(f"Optimal resume length ({word_count} words).")
        elif word_count < 250:
            score += 4
            issues.append(f"Resume text is too brief ({word_count} words).")
            suggestions.append("Expand on project descriptions, duties, and technical competencies.")
        else:
            score += 6
            issues.append(f"Resume is very lengthy ({word_count} words). Consider concise formatting.")

        # Cap score
        final_ats_score = min(max(score, 10), 100)

        return {
            "ats_score": final_ats_score,
            "strengths": strengths,
            "issues": issues,
            "suggestions": suggestions,
            "word_count": word_count,
            "detected_headings_count": len(detected_headings),
        }


# Global instance
ats_analyzer = ATSAnalyzer()
