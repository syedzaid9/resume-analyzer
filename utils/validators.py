"""Input and File Validators for ResumeAI.
"""

from pathlib import Path
from typing import Tuple, Optional
from src.config import MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS, MIN_RESUME_WORD_COUNT


def validate_file_upload(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validates uploaded Streamlit file object for type and size constraints."""
    if uploaded_file is None:
        return False, "No file uploaded. Please select a PDF resume."

    filename = getattr(uploaded_file, "name", "")
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file format '{ext}'. ResumeAI currently accepts PDF documents (.pdf).",
        )

    # Size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        return (
            False,
            f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    return True, None


def validate_extracted_text(text: str) -> Tuple[bool, Optional[str]]:
    """Validates extracted resume text length and readability."""
    if not text or not isinstance(text, str):
        return False, "Unable to extract text from document. Please ensure it is a valid text-based PDF."

    words = text.split()
    if len(words) < MIN_RESUME_WORD_COUNT:
        return (
            False,
            f"Extracted text contains only {len(words)} words. "
            "Please upload a comprehensive resume or a non-scanned text PDF.",
        )

    return True, None
