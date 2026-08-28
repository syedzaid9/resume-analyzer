"""Resume Parser Module for ResumeAI.
Extracts candidate personal information, education, experience history, projects,
and quantitative metrics from parsed resume text without fabricating data.
"""

import re
from typing import Dict, List, Any, Optional
from src.config import POWER_ACTION_VERBS
from src.section_extractor import section_extractor


class ResumeParser:
    """Extracts candidate profile attributes and structural metadata from resumes."""

    # Regex patterns for personal contact info
    EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    PHONE_REGEX = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    LINKEDIN_REGEX = r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|profile)\/([A-Za-z0-9_-]+)"
    GITHUB_REGEX = r"(?:https?:\/\/)?(?:www\.)?github\.com\/([A-Za-z0-9_-]+)"
    PORTFOLIO_REGEX = r"(?:https?:\/\/)?(?:www\.)?[A-Za-z0-9_-]+\.(?:io|dev|me|com|org)(?:\/[A-Za-z0-9_-]*)*"

    # Education degree markers
    DEGREE_PATTERNS = [
        r"\b(ph\.?d|doctorate|doctor of philosophy)\b",
        r"\b(master(?:'s)? of science|m\.?s\.?|master of technology|m\.?tech|m\.?eng|m\.?b\.?a|master of business administration)\b",
        r"\b(bachelor(?:'s)? of science|b\.?s\.?|bachelor of technology|b\.?tech|b\.?e\.?|bachelor of engineering|b\.?a\.?|bachelor of arts|b\.?c\.?a)\b",
        r"\b(associate(?:'s)? degree|diploma)\b",
    ]

    COMMON_MAJORS = [
        "Computer Science", "Information Technology", "Data Science", "Software Engineering",
        "Electrical Engineering", "Statistics", "Mathematics", "Artificial Intelligence",
        "Business Analytics", "Cybersecurity", "Computer Applications", "Mechanical Engineering",
        "Economics", "Finance", "Marketing", "Human Resources", "Management"
    ]

    COMMON_TITLES = [
        "Software Engineer", "Senior Software Engineer", "Full Stack Developer", "Frontend Developer",
        "Backend Developer", "Data Scientist", "Senior Data Scientist", "Data Analyst",
        "Machine Learning Engineer", "AI Engineer", "DevOps Engineer", "Cloud Architect",
        "Cybersecurity Analyst", "Product Manager", "Project Manager", "Scrum Master",
        "UI/UX Designer", "Database Administrator", "System Administrator", "Network Engineer",
        "Intern", "Software Developer Intern", "Research Assistant", "Graduate Assistant"
    ]

    def extract_personal_info(self, text: str, header_text: str = "") -> Dict[str, str]:
        """Extracts candidate contact information."""
        info = {
            "name": "Not detected",
            "email": "Not detected",
            "phone": "Not detected",
            "linkedin": "Not detected",
            "github": "Not detected",
            "portfolio": "Not detected",
        }

        # 1. Email
        email_match = re.search(self.EMAIL_REGEX, text)
        if email_match:
            info["email"] = email_match.group(0)

        # 2. Phone
        phone_match = re.search(self.PHONE_REGEX, text)
        if phone_match:
            info["phone"] = phone_match.group(0).strip()

        # 3. LinkedIn
        linkedin_match = re.search(self.LINKEDIN_REGEX, text, re.IGNORECASE)
        if linkedin_match:
            info["linkedin"] = f"https://linkedin.com/in/{linkedin_match.group(1)}"

        # 4. GitHub
        github_match = re.search(self.GITHUB_REGEX, text, re.IGNORECASE)
        if github_match:
            info["github"] = f"https://github.com/{github_match.group(1)}"

        # 5. Portfolio
        portfolio_matches = re.finditer(r"(?:https?:\/\/)?([a-zA-Z0-9-]+\.(?:io|dev|me|site))\b", text, re.IGNORECASE)
        for pm in portfolio_matches:
            domain = pm.group(0).lower()
            if "linkedin" not in domain and "github" not in domain and "gmail" not in domain:
                info["portfolio"] = domain
                break

        # 6. Candidate Name Heuristic: First clean non-empty line of the header that is not email/phone
        search_block = header_text if header_text else text
        lines = [line.strip() for line in search_block.split("\n") if line.strip()]
        for line in lines[:5]:
            # Skip if contains email, URL, phone, or generic heading
            if re.search(self.EMAIL_REGEX, line) or re.search(self.PHONE_REGEX, line) or "resume" in line.lower() or "curriculum" in line.lower():
                continue
            words = line.split()
            # A plausible name is usually 2 to 4 words with alphabetic chars
            if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
                info["name"] = line
                break

        return info

    def extract_education(self, text: str, education_section: str = "") -> Dict[str, Any]:
        """Parses degrees, majors, and graduation years."""
        target_text = education_section if education_section else text
        detected_degrees = []
        detected_majors = []
        grad_years = []

        # Find degree mentions
        for pattern in self.DEGREE_PATTERNS:
            matches = re.finditer(pattern, target_text, re.IGNORECASE)
            for m in matches:
                deg = m.group(0).strip()
                if deg not in detected_degrees:
                    detected_degrees.append(deg.title())

        # Find major / field of study
        for major in self.COMMON_MAJORS:
            if re.search(r"\b" + re.escape(major) + r"\b", target_text, re.IGNORECASE):
                if major not in detected_majors:
                    detected_majors.append(major)

        # Find 4-digit years (e.g. 2015 - 2024)
        year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", target_text)
        if year_matches:
            grad_years = sorted(list(set(year_matches)))

        return {
            "degrees": detected_degrees if detected_degrees else ["Not detected"],
            "majors": detected_majors if detected_majors else ["Not detected"],
            "grad_years": grad_years if grad_years else ["Not detected"],
            "has_degree": len(detected_degrees) > 0,
            "raw_education_text": education_section.strip() if education_section else "Not detected",
        }

    def extract_experience(self, text: str, experience_section: str = "") -> Dict[str, Any]:
        """Analyzes work history, action verbs, and quantifiable metrics."""
        target_text = experience_section if experience_section else text

        # 1. Job Titles
        detected_titles = []
        for title in self.COMMON_TITLES:
            if re.search(r"\b" + re.escape(title) + r"\b", target_text, re.IGNORECASE):
                if title not in detected_titles:
                    detected_titles.append(title)

        # 2. Action Verbs
        found_action_verbs = []
        text_lower = target_text.lower()
        for verb in POWER_ACTION_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", text_lower):
                found_action_verbs.append(verb)

        # 3. Measurable Metrics (Percentages, Dollar amounts, Big Numbers, Multipliers)
        metrics = re.findall(
            r"(\b\d+(?:\.\d+)?%\b|\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[kKmMbB])?|\b\d+\+?\s*(?:users|clients|teams|projects|microservices|models|x)\b)",
            target_text
        )

        # 4. Estimated years of experience mention (e.g., "5+ years experience")
        years_exp_match = re.search(r"\b(\d+)\+?\s*years?(?:\s+of)?\s+experience\b", text, re.IGNORECASE)
        est_years = int(years_exp_match.group(1)) if years_exp_match else None

        return {
            "detected_titles": detected_titles if detected_titles else ["Not detected"],
            "action_verbs_count": len(found_action_verbs),
            "action_verbs_sample": found_action_verbs[:10],
            "metrics_count": len(metrics),
            "metrics_sample": [m.strip() for m in metrics[:5]],
            "estimated_years": est_years if est_years is not None else "Not explicitly stated",
            "has_experience_section": len(experience_section.strip()) > 30,
        }

    def extract_projects(self, text: str, projects_section: str = "") -> Dict[str, Any]:
        """Analyzes project portfolio and metrics."""
        target_text = projects_section if projects_section else text
        has_projects = len(projects_section.strip()) > 20

        # Look for project bullets or lines
        lines = [l.strip() for l in target_text.split("\n") if len(l.strip()) > 15]
        
        metrics_in_projects = re.findall(r"(\b\d+%\b|\$\d+|\b\d+\s*x\b)", target_text)

        return {
            "has_projects_section": has_projects,
            "project_lines_count": len(lines),
            "project_metrics_detected": len(metrics_in_projects) > 0,
            "project_metrics_sample": metrics_in_projects[:5],
            "raw_projects_text": projects_section.strip() if projects_section else "Not detected",
        }

    def parse_full_resume(self, text: str) -> Dict[str, Any]:
        """Performs complete information extraction across all resume dimensions."""
        sections = section_extractor.extract_sections(text)

        personal_info = self.extract_personal_info(text, sections.get("contact", ""))
        education_info = self.extract_education(text, sections.get("education", ""))
        experience_info = self.extract_experience(text, sections.get("experience", ""))
        projects_info = self.extract_projects(text, sections.get("projects", ""))

        return {
            "personal_info": personal_info,
            "education": education_info,
            "experience": experience_info,
            "projects": projects_info,
            "sections": sections,
        }


# Global instance
resume_parser = ResumeParser()
