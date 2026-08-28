"""Central Configuration Module for ResumeAI.
Defines all directory paths, model weights, target domains, ATS rules, and UI color palettes.
Avoids hardcoding constants throughout the application.
"""

from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. Base Directories & Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOMAIN_REQUIREMENTS_DIR = DATA_DIR / "domain_requirements"

MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_DIR = MODELS_DIR / "classifier"
VECTORIZER_DIR = MODELS_DIR / "vectorizer"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

BEST_CLASSIFIER_PATH = CLASSIFIER_DIR / "best_classifier.joblib"
TFIDF_VECTORIZER_PATH = VECTORIZER_DIR / "tfidf_vectorizer.joblib"

RAW_DATASET_PATH = RAW_DATA_DIR / "resume_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "cleaned_resumes.csv"

OCCUPATIONS_CSV_PATH = DOMAIN_REQUIREMENTS_DIR / "occupations.csv"
SKILLS_CSV_PATH = DOMAIN_REQUIREMENTS_DIR / "skills.csv"
TECHNOLOGIES_CSV_PATH = DOMAIN_REQUIREMENTS_DIR / "technologies.csv"
DOMAIN_MAPPING_CSV_PATH = DOMAIN_REQUIREMENTS_DIR / "domain_mapping.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Supported Target Domains (22 Domains)
# ==========================================
SUPPORTED_DOMAINS: List[str] = [
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Software Development",
    "Python Developer",
    "Java Developer",
    "Web Development",
    "Frontend Development",
    "Backend Development",
    "Full Stack Development",
    "Data Analytics",
    "Database Administration",
    "Cyber Security",
    "Cloud Computing",
    "DevOps",
    "Networking",
    "UI/UX Design",
    "Digital Marketing",
    "Finance",
    "HR",
    "Business Development",
    "Project Management",
]

# ==========================================
# 3. Transparent Resume Scoring Engine Weights
# (Configurable - Total Must Equal 100%)
# ==========================================
SCORING_WEIGHTS: Dict[str, float] = {
    "skill_match": 30.0,       # 30% Skill Alignment
    "experience": 20.0,        # 20% Experience & Role History
    "projects": 15.0,          # 15% Project Portfolio & Technical Depth
    "education": 10.0,         # 10% Educational Relevance
    "ats_compatibility": 10.0, # 10% ATS Standards & Parsing Compliance
    "structure": 10.0,         # 10% Section Organization & Completeness
    "formatting": 5.0,         # 5% Formatting & Readability Standards
}

# Verification check for scoring weights
assert (
    abs(sum(SCORING_WEIGHTS.values()) - 100.0) < 1e-4
), "SCORING_WEIGHTS must sum up to exactly 100%!"

# ==========================================
# 4. Rating Tiers
# ==========================================
RATING_TIERS: List[Dict[str, Any]] = [
    {"min": 90, "max": 100, "label": "Excellent", "color": "#16A34A", "badge": "🌟 Top Tier Match"},
    {"min": 80, "max": 89, "label": "Very Good", "color": "#2563EB", "badge": "✨ Strong Candidate"},
    {"min": 70, "max": 79, "label": "Good", "color": "#7C3AED", "badge": "👍 Solid Baseline"},
    {"min": 60, "max": 69, "label": "Needs Improvement", "color": "#F59E0B", "badge": "⚠️ Action Required"},
    {"min": 0, "max": 59, "label": "Needs Improvement", "color": "#DC2626", "badge": "🚨 Critical Gaps"},
]

# ==========================================
# 5. File Upload & Parsing Limits
# ==========================================
MAX_UPLOAD_SIZE_MB: int = 10
ALLOWED_EXTENSIONS: List[str] = [".pdf"]
MIN_RESUME_WORD_COUNT: int = 40
MAX_RESUME_WORD_COUNT: int = 2500

# ==========================================
# 6. Skill Categories & Importance Levels
# ==========================================
SKILL_CATEGORIES: List[str] = [
    "Programming Languages",
    "Frameworks & Libraries",
    "Databases",
    "Cloud & DevOps",
    "AI / Machine Learning",
    "Tools & Platforms",
    "Soft Skills",
]

IMPORTANCE_LEVELS: Dict[str, Dict[str, Any]] = {
    "Critical": {"weight": 1.0, "badge_color": "#DC2626", "priority": 1},
    "High": {"weight": 0.75, "badge_color": "#F59E0B", "priority": 2},
    "Medium": {"weight": 0.5, "badge_color": "#2563EB", "priority": 3},
    "Low": {"weight": 0.25, "badge_color": "#64748B", "priority": 4},
}

# ==========================================
# 7. UI Design Tokens & Semantics
# ==========================================
UI_THEME = {
    "primary": "#2563EB",       # Royal Blue
    "secondary": "#7C3AED",     # Vibrant Purple
    "success": "#16A34A",       # Forest Green
    "warning": "#F59E0B",       # Amber Orange
    "danger": "#DC2626",        # Crimson Red
    "background": "#F8FAFC",    # Slate 50
    "card_bg": "#FFFFFF",       # Pure White
    "text_primary": "#0F172A",  # Slate 900
    "text_muted": "#64748B",    # Slate 500
    "border": "#E2E8F0",        # Slate 200
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
}

# ==========================================
# 8. Action Verbs for Resume Quality Audit
# ==========================================
POWER_ACTION_VERBS: List[str] = [
    "accelerated", "achieved", "administered", "architected", "automated",
    "built", "calculated", "championed", "collaborated", "constructed",
    "created", "debugged", "decreased", "delivered", "deployed", "designed",
    "developed", "devised", "directed", "doubled", "engineered", "established",
    "evaluated", "expanded", "expedited", "formulated", "founded", "generated",
    "guided", "implemented", "improved", "increased", "initiated", "innovated",
    "installed", "instituted", "integrated", "invented", "launched", "lead",
    "maximized", "mentored", "minimized", "modeled", "modernized", "negotiated",
    "orchestrated", "optimized", "overhauled", "pioneered", "produced",
    "programmed", "reduced", "refactored", "resolved", "restructured",
    "revamped", "scaled", "simplified", "spearheaded", "standardized",
    "streamlined", "strengthened", "surpassed", "transformed", "upgraded"
]
