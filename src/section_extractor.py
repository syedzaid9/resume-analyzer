"""Section Extractor Module for ResumeAI.
Segments unstructured resume text into standardized structural sections:
Summary, Education, Experience, Projects, Skills, and Certifications.
"""

import re
from typing import Dict, List, Tuple


class SectionExtractor:
    """Extracts structured sections from raw resume text using regex headers."""

    SECTION_PATTERNS = {
        "summary": [
            r"\b(professional\s+summary|executive\s+summary|summary|profile|about\s+me|career\s+objective|objective)\b"
        ],
        "education": [
            r"\b(education|academic\s+background|academic\s+qualifications|qualifications|degrees?|educational\s+history)\b"
        ],
        "experience": [
            r"\b(work\s+experience|professional\s+experience|experience|employment\s+history|work\s+history|career\s+history)\b"
        ],
        "projects": [
            r"\b(projects|key\s+projects|academic\s+projects|personal\s+projects|technical\s+projects|portfolio\s+projects)\b"
        ],
        "skills": [
            r"\b(skills|technical\s+skills|core\s+competencies|technologies|skills\s+and\s+tools|areas\s+of\s+expertise|programming\s+skills)\b"
        ],
        "certifications": [
            r"\b(certifications|certificates|licenses|awards|honors|achievements|accreditations)\b"
        ],
    }

    def __init__(self):
        pass

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Splits full resume text into standard section segments."""
        if not text:
            return {sec: "" for sec in self.SECTION_PATTERNS}

        lines = [line.strip() for line in text.split("\n")]
        
        # Identify heading candidate lines: short lines that match section keywords
        heading_matches: List[Tuple[int, str]] = [] # (line_index, section_name)

        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            if not line_clean or len(line_clean) > 40:
                continue

            # Check if this line matches any section heading pattern
            for sec_name, patterns in self.SECTION_PATTERNS.items():
                for pat in patterns:
                    # Match if heading is by itself or with colon/divider
                    if re.search(r"^[\s#\*\-\>]*" + pat + r"[\s:\-\|\*]*$", line_clean):
                        heading_matches.append((i, sec_name))
                        break

        # If no explicit headings found, fallback to heuristic keyword search
        sections: Dict[str, str] = {sec: "" for sec in self.SECTION_PATTERNS}
        sections["full_text"] = text

        if not heading_matches:
            # Fallback: simple keyword containment chunks
            return sections

        # Sort matches by line index
        heading_matches.sort(key=lambda x: x[0])

        # Header/Contact info is everything before the first heading
        first_heading_idx = heading_matches[0][0]
        sections["contact"] = "\n".join(lines[:first_heading_idx]).strip()

        # Slice between consecutive headings
        for idx in range(len(heading_matches)):
            start_line, sec_name = heading_matches[idx]
            end_line = heading_matches[idx + 1][0] if idx + 1 < len(heading_matches) else len(lines)
            sec_content = "\n".join(lines[start_line + 1 : end_line]).strip()
            
            # If section already has content, append
            if sections[sec_name]:
                sections[sec_name] += "\n" + sec_content
            else:
                sections[sec_name] = sec_content

        return sections


# Global instance
section_extractor = SectionExtractor()
