"""Text Preprocessor Module for ResumeAI.
Performs text normalization, HTML stripping, whitespace cleanup, and technical-token preservation.
Crucially protects programming tokens such as C++, C#, .NET, Node.js, React.js, CI/CD, etc.
"""

import re
import html
from typing import List, Optional
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources
try:
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    try:
        _STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        _STOP_WORDS = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "were", "of", "by", "as", "from", "it", "this", "that"
        }

# Technical terms mapping to temporary safe alphanumeric placeholders
TECH_TOKEN_MAP = [
    (r"(?i)(?:\b|(?<=\s))c\+\+(?=\b|\s|[,\.;]|$)", "techcplusplus"),
    (r"(?i)(?:\b|(?<=\s))c#(?=\b|\s|[,\.;]|$)", "techcsharp"),
    (r"(?i)(?:\b|(?<=\s))\.net(?=\b|\s|[,\.;]|$)", "techdotnet"),
    (r"(?i)(?:\b|(?<=\s))asp\.net(?=\b|\s|[,\.;]|$)", "techaspdotnet"),
    (r"(?i)\bnode\.js\b", "technodejs"),
    (r"(?i)\breact\.js\b", "techreactjs"),
    (r"(?i)\bnext\.js\b", "technextjs"),
    (r"(?i)\bvue\.js\b", "techvuejs"),
    (r"(?i)\bci/cd\b", "techcicd"),
    (r"(?i)\btcp/ip\b", "techtcpip"),
    (r"(?i)\bpower\s+bi\b", "techpowerbi"),
    (r"(?i)\bscikit-learn\b", "techscikitlearn"),
]

TECH_TOKEN_RESTORE = {
    "techcplusplus": "c++",
    "techcsharp": "c#",
    "techdotnet": ".net",
    "techaspdotnet": "asp.net",
    "technodejs": "node.js",
    "techreactjs": "react.js",
    "technextjs": "next.js",
    "techvuejs": "vue.js",
    "techcicd": "ci/cd",
    "techtcpip": "tcp/ip",
    "techpowerbi": "power bi",
    "techscikitlearn": "scikit-learn",
}


class TextPreprocessor:
    """Robust text cleaning pipeline preserving technical domain tokens."""

    def __init__(self, remove_stopwords: bool = False, preserve_case: bool = False):
        self.remove_stopwords = remove_stopwords
        self.preserve_case = preserve_case

    def clean_html(self, text: str) -> str:
        """Unescapes HTML entities and strips HTML markup."""
        if not text:
            return ""
        text = html.unescape(text)
        clean_re = re.compile(r"<[^>]+>")
        return clean_re.sub(" ", text)

    def normalize_unicode(self, text: str) -> str:
        """Converts special quotes, dashes, and non-breaking spaces into standard ASCII."""
        if not text:
            return ""
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2022": " • ", "\u25aa": " • ",
            "\u00a0": " ", "\t": " ", "\r": "\n"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def clean_text_for_ml(self, text: str) -> str:
        """Performs comprehensive normalization for TF-IDF vectorization and ML classification.
        Preserves C++, C#, .NET, Node.js while removing punctuation noise and standard stopwords.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Clean HTML & normalize unicode
        text = self.clean_html(text)
        text = self.normalize_unicode(text)

        # 2. Lowercase
        text_lower = " " + text.lower() + " "

        # 3. Protect technical tokens
        for pattern, placeholder in TECH_TOKEN_MAP:
            text_lower = re.sub(pattern, f" {placeholder} ", text_lower)

        # 4. Remove URL links, emails, and phone numbers
        text_lower = re.sub(r"https?://\S+|www\.\S+", " ", text_lower)
        text_lower = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text_lower)
        text_lower = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", text_lower)

        # 5. Remove unwanted punctuation while keeping alphanumeric and underscore
        text_lower = re.sub(r"[^\w\s]", " ", text_lower)

        # 6. Restore technical tokens
        words = text_lower.split()
        restored_words = []
        for w in words:
            if w in TECH_TOKEN_RESTORE:
                restored_words.append(TECH_TOKEN_RESTORE[w])
            else:
                if self.remove_stopwords:
                    if w not in _STOP_WORDS and len(w) > 1:
                        restored_words.append(w)
                else:
                    restored_words.append(w)

        # 7. Collapse whitespace
        return " ".join(restored_words)

    def extract_clean_sentences(self, text: str) -> List[str]:
        """Extracts bullet lines / sentences cleanly for section & experience analysis."""
        if not text:
            return []
        text = self.clean_html(text)
        text = self.normalize_unicode(text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\s•\-\*\>\d+\.]+", "", line).strip()
            if len(line) > 5:
                cleaned.append(line)
        return cleaned


# Global convenient instance
preprocessor = TextPreprocessor()
