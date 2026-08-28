"""Skill Extractor Module for ResumeAI.
Extracts and categorizes technical and soft skills from resume text using a comprehensive
domain knowledge base, synonym normalization, and boundary-aware regex matching.
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import pandas as pd

from src.config import SKILLS_CSV_PATH, TECHNOLOGIES_CSV_PATH, SKILL_CATEGORIES


class SkillExtractor:
    """Extracts, normalizes, and categorizes skills from resume text."""

    def __init__(self):
        self.skills_db: Dict[str, Dict[str, Any]] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.category_skills: Dict[str, List[str]] = {cat: [] for cat in SKILL_CATEGORIES}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads canonical skills and alias mapping from domain knowledge files."""
        if not SKILLS_CSV_PATH.exists():
            print(f"[WARN] Skills knowledge base not found at {SKILLS_CSV_PATH}")
            return

        df_skills = pd.read_csv(SKILLS_CSV_PATH)
        for _, row in df_skills.iterrows():
            canonical = str(row["skill_name"]).strip()
            category = str(row.get("category", "Tools & Platforms")).strip()
            synonyms = str(row.get("synonyms", "")).strip()

            if category not in self.category_skills:
                self.category_skills[category] = []
            self.category_skills[category].append(canonical)

            self.skills_db[canonical] = {
                "category": category,
                "description": str(row.get("description", "")),
                "synonyms": [s.strip() for s in synonyms.split("|") if s.strip()],
            }

            # Map canonical name to itself (lowercased)
            self.alias_to_canonical[canonical.lower()] = canonical

            # Map all synonyms
            for syn in self.skills_db[canonical]["synonyms"]:
                self.alias_to_canonical[syn.lower()] = canonical

        # Load additional technology aliases if available
        if TECHNOLOGIES_CSV_PATH.exists():
            df_tech = pd.read_csv(TECHNOLOGIES_CSV_PATH)
            for _, row in df_tech.iterrows():
                tech_name = str(row["tech_name"]).strip()
                if tech_name.lower() not in self.alias_to_canonical:
                    self.alias_to_canonical[tech_name.lower()] = tech_name

    def extract_skills(self, text: str) -> Dict[str, Any]:
        """Extracts all matching skills from text, returns structured dictionary with categories."""
        if not text:
            return {
                "all_detected_skills": [],
                "categorized_skills": {cat: [] for cat in SKILL_CATEGORIES},
                "total_skills_count": 0,
            }

        text_lower = " " + text.lower() + " "
        detected_canonical: Set[str] = set()

        # 1. Match specific technical tokens that need special boundary handling
        special_tokens = {
            r"\bc\+\+\b": "C++",
            r"\bc#\b": "C#",
            r"\b\.net\b": ".NET Core",
            r"\basp\.net\b": ".NET Core",
            r"\bnode\.?js\b": "Node.js",
            r"\breact\.?js\b": "React",
            r"\bnext\.?js\b": "Next.js",
            r"\bvue\.?js\b": "Vue.js",
            r"\bci/cd\b": "CI/CD",
            r"\btcp/ip\b": "Networking",
            r"\bpower\s+bi\b": "Power BI",
            r"\bscikit-learn\b": "Scikit-learn",
            r"\bpostgre\s*sql\b|\bpostgres\b|\bpsql\b": "PostgreSQL",
            r"\bmy\s*sql\b": "MySQL",
            r"\bms\s*sql\b|\bsql\s*server\b": "Microsoft SQL Server",
            r"\bui/ux\b": "UI/UX Design",
            r"\br\s+programming\b|\br\s+language\b": "R",
        }

        for pattern, canon in special_tokens.items():
            if re.search(pattern, text_lower):
                detected_canonical.add(canon)

        # 2. Match general aliases with regex word boundary
        for alias, canonical in self.alias_to_canonical.items():
            # Skip very short aliases that cause false positives (e.g. 'c', 'r', 'go', 'ai', 'it', 'me')
            if len(alias) <= 2:
                # Require explicit capitalization or surrounding context in original text
                escaped = re.escape(alias)
                # Check for standalone uppercase token in original text
                if re.search(r"\b" + escaped.upper() + r"\b", text):
                    detected_canonical.add(canonical)
                continue

            # Standard word boundary match
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                detected_canonical.add(canonical)

        # 3. Categorize detected skills
        categorized: Dict[str, List[str]] = {cat: [] for cat in SKILL_CATEGORIES}

        for skill in sorted(list(detected_canonical)):
            cat = self.skills_db.get(skill, {}).get("category", "Tools & Platforms")
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(skill)

        # Clean empty categories if desired or keep standard
        return {
            "all_detected_skills": sorted(list(detected_canonical)),
            "categorized_skills": categorized,
            "total_skills_count": len(detected_canonical),
        }


# Global instance
skill_extractor = SkillExtractor()
