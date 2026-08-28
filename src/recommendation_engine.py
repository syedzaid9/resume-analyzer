"""Recommendation Engine for ResumeAI.
Generates evidence-based strengths, weaknesses, prioritized learning roadmaps,
actionable bullet enhancements, project build recommendations, and cross-domain career matches.
Strictly avoids fabricating achievements.
"""

from typing import Dict, List, Any
import pandas as pd

from src.config import OCCUPATIONS_CSV_PATH


class RecommendationEngine:
    """Produces tailored, actionable resume improvements and portfolio project ideas."""

    PROJECT_TEMPLATES = [
        {
            "skills": {"docker", "fastapi", "aws", "kubernetes", "model deployment"},
            "title": "Production AI/ML Microservice Deployment",
            "description": "Develop and containerize a high-performance machine learning inference API using FastAPI, Docker, and deploy it to AWS ECS or Kubernetes with automated CI/CD.",
            "difficulty": "Advanced",
        },
        {
            "skills": {"react", "node.js", "typescript", "rest api", "postgresql"},
            "title": "Full-Stack Analytics & Job Portal",
            "description": "Construct a responsive single-page web dashboard using React, TypeScript, and Node.js connected to a PostgreSQL database with JWT authentication.",
            "difficulty": "Intermediate",
        },
        {
            "skills": {"sql", "power bi", "tableau", "data modeling", "snowflake"},
            "title": "Enterprise Business Intelligence & KPI Dashboard",
            "description": "Architect a centralized data warehouse in Snowflake or PostgreSQL and build interactive executive dashboards in Power BI with custom DAX measures.",
            "difficulty": "Intermediate",
        },
        {
            "skills": {"terraform", "ci/cd", "github actions", "aws", "prometheus"},
            "title": "Automated Cloud Infrastructure & GitOps Pipeline",
            "description": "Provision reproducible AWS cloud infrastructure using Terraform (IaC) and configure GitHub Actions pipelines with Prometheus/Grafana observability.",
            "difficulty": "Advanced",
        },
        {
            "skills": {"deep learning", "pytorch", "transformers", "llm", "langchain"},
            "title": "Retrieval-Augmented Generation (RAG) Document Agent",
            "description": "Build an intelligent semantic search agent leveraging PyTorch, LangChain, and vector embeddings to query internal technical documentation.",
            "difficulty": "Advanced",
        },
        {
            "skills": {"figma", "ui/ux design", "html5", "tailwind css"},
            "title": "Design System & Accessible Component Library",
            "description": "Create an atomic design system in Figma and implement responsive, WCAG-compliant UI components using Tailwind CSS.",
            "difficulty": "Beginner/Intermediate",
        },
    ]

    def __init__(self):
        pass

    def generate_strengths(
        self,
        parsed_data: Dict[str, Any],
        matched_skills: List[Dict[str, Any]],
        ats_data: Dict[str, Any],
        target_domain: str,
    ) -> List[str]:
        """Derives evidence-based strengths directly from extracted resume data."""
        strengths: List[str] = []

        # 1. Skill strengths
        if len(matched_skills) >= 5:
            top_skills = [s["skill_name"] for s in matched_skills[:4]]
            strengths.append(f"Strong alignment in core {target_domain} skills: {', '.join(top_skills)}.")
        elif len(matched_skills) > 0:
            top_skills = [s["skill_name"] for s in matched_skills]
            strengths.append(f"Demonstrated foundational competencies in {', '.join(top_skills)}.")

        # 2. Education strength
        education = parsed_data.get("education", {})
        if education.get("has_degree"):
            deg = education["degrees"][0]
            major = education["majors"][0] if education.get("majors") and education["majors"][0] != "Not detected" else "quantitative discipline"
            strengths.append(f"Strong academic foundation with a {deg} in {major}.")

        # 3. Experience & Action Verbs
        experience = parsed_data.get("experience", {})
        if experience.get("action_verbs_count", 0) >= 4:
            sample_verbs = experience.get("action_verbs_sample", [])[:3]
            strengths.append(f"Action-driven impact statements utilizing power verbs ('{', '.join(sample_verbs)}').")

        # 4. Metrics
        if experience.get("metrics_count", 0) >= 1:
            strengths.append("Contains quantified business metrics and measurable outcome indicators.")

        # 5. ATS
        if ats_data.get("ats_score", 0) >= 80:
            strengths.append(f"High ATS parsing compatibility score ({ats_data.get('ats_score')}/100).")

        return strengths if strengths else ["Solid foundational resume structure with baseline technical competencies."]

    def generate_weaknesses(
        self,
        parsed_data: Dict[str, Any],
        missing_skills: List[Dict[str, Any]],
        ats_data: Dict[str, Any],
        target_domain: str,
    ) -> List[str]:
        """Identifies genuine gaps and areas of vulnerability."""
        weaknesses: List[str] = []

        # 1. Critical Missing Skills
        critical_missing = [s["skill_name"] for s in missing_skills if s.get("importance") == "Critical"]
        if critical_missing:
            weaknesses.append(f"Missing high-priority critical skills for {target_domain}: {', '.join(critical_missing[:4])}.")

        # 2. Measurable Metrics
        experience = parsed_data.get("experience", {})
        if experience.get("metrics_count", 0) == 0:
            weaknesses.append("Experience bullets lack measurable numbers, percentages, or verified performance metrics.")

        # 3. Action Verbs
        if experience.get("action_verbs_count", 0) < 3:
            weaknesses.append("Work history descriptions rely heavily on passive duties rather than proactive action verbs.")

        # 4. ATS Issues
        personal_info = parsed_data.get("personal_info", {})
        if personal_info.get("linkedin") == "Not detected" and personal_info.get("github") == "Not detected":
            weaknesses.append("No professional profile links (LinkedIn, GitHub, or online portfolio) detected.")

        return weaknesses if weaknesses else ["Minor optimizations possible in keyword density and bullet phrasing."]

    def generate_project_recommendations(
        self, missing_skills: List[Dict[str, Any]], target_domain: str
    ) -> List[Dict[str, Any]]:
        """Recommends specific portfolio projects that bridge missing skill gaps."""
        missing_names_lower = {s["skill_name"].lower() for s in missing_skills}
        recommended_projects = []

        for template in self.PROJECT_TEMPLATES:
            # Check overlap between missing skills and project skills
            overlap = template["skills"].intersection(missing_names_lower)
            if overlap or len(recommended_projects) < 2:
                recommended_projects.append({
                    "title": template["title"],
                    "description": template["description"],
                    "difficulty": template["difficulty"],
                    "skills_addressed": [s.title() for s in template["skills"] if s in missing_names_lower] or [s.title() for s in list(template["skills"])[:3]],
                })

        return recommended_projects[:3]

    def generate_actionable_improvements(
        self,
        parsed_data: Dict[str, Any],
        missing_skills: List[Dict[str, Any]],
        ats_data: Dict[str, Any],
        target_domain: str,
    ) -> List[Dict[str, str]]:
        """Generates concrete, step-by-step guidance for resume improvement."""
        improvements = []

        # 1. Critical Skill Advice
        critical_missing = [s["skill_name"] for s in missing_skills if s.get("importance") == "Critical"]
        if critical_missing:
            improvements.append({
                "category": "Skill Enhancement",
                "issue": f"Target domain ({target_domain}) prioritizes {', '.join(critical_missing[:3])}.",
                "recommendation": f"Acquire or highlight hands-on experience in {', '.join(critical_missing[:3])}. Add relevant coursework or practical projects demonstrating competence.",
            })

        # 2. Quantified Metrics
        improvements.append({
            "category": "Quantified Results",
            "issue": "Recruiters favor measurable outcomes over task lists.",
            "recommendation": "Rewrite bullet points using the Google XYZ Formula: 'Accomplished [X] as measured by [Y], by doing [Z]'. Example: 'Optimized database indexing, reducing query response times by 35%'.",
        })

        # 3. Action Verbs
        experience = parsed_data.get("experience", {})
        if experience.get("action_verbs_count", 0) < 4:
            improvements.append({
                "category": "Action Verbs",
                "issue": "Phrasing can be made more impactful.",
                "recommendation": "Replace phrases like 'responsible for building' with powerful verbs such as 'Engineered', 'Architected', 'Automated', or 'Pioneered'.",
            })

        # 4. ATS Optimization
        if ats_data.get("issues"):
            top_issue = ats_data["issues"][0]
            improvements.append({
                "category": "ATS Parsing",
                "issue": top_issue,
                "recommendation": "Ensure your resume adheres to clean single-column formatting, standard section headings, and explicit contact information.",
            })

        return improvements


# Global instance
recommendation_engine = RecommendationEngine()
