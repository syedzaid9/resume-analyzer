"""PDF Parser Module for ResumeAI.
Extracts clean text and metadata from PDF files using PyMuPDF (fitz) with pypdf fallback.
Validates file integrity, empty documents, corrupted files, and scanned/image-only PDFs.
"""

import io
from pathlib import Path
from typing import Dict, Any, Union, BinaryIO

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class PDFParser:
    """Robust PDF text extractor with fallback and OCR/scanned detection."""

    def __init__(self, min_words_threshold: int = 30):
        self.min_words_threshold = min_words_threshold

    def parse_pdf(self, file_source: Union[str, Path, bytes, BinaryIO]) -> Dict[str, Any]:
        """Parses a PDF file from filepath, bytes, or file-like object.
        Returns a structured dictionary containing text, page count, word count, status, and errors.
        """
        result = {
            "success": False,
            "text": "",
            "raw_text": "",
            "pages": [],
            "page_count": 0,
            "word_count": 0,
            "char_count": 0,
            "is_scanned": False,
            "error": None,
            "warning": None,
            "parser_used": None,
        }

        # 1. Read bytes
        pdf_bytes = b""
        try:
            if isinstance(file_source, (str, Path)):
                path = Path(file_source)
                if not path.exists():
                    result["error"] = f"File not found at: {path}"
                    return result
                if path.stat().st_size == 0:
                    result["error"] = "The uploaded PDF file is empty (0 bytes)."
                    return result
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
            elif isinstance(file_source, bytes):
                pdf_bytes = file_source
            elif hasattr(file_source, "read"):
                # File-like object (e.g. Streamlit UploadedFile)
                file_source.seek(0)
                pdf_bytes = file_source.read()
                file_source.seek(0)
            else:
                result["error"] = "Invalid PDF file source provided."
                return result

            if not pdf_bytes or len(pdf_bytes) == 0:
                result["error"] = "The uploaded PDF file contains no data."
                return result

        except Exception as e:
            result["error"] = f"Error reading PDF file stream: {str(e)}"
            return result

        # 2. Try PyMuPDF (fitz) first
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                result["page_count"] = len(doc)
                pages_text = []
                for i in range(len(doc)):
                    page = doc[i]
                    p_text = page.get_text("text") or ""
                    pages_text.append(p_text)

                doc.close()
                result["pages"] = pages_text
                result["raw_text"] = "\n\n".join(pages_text).strip()
                result["text"] = result["raw_text"]
                result["parser_used"] = "PyMuPDF"
                result["success"] = True
            except Exception as e:
                result["warning"] = f"PyMuPDF failed ({str(e)}), attempting pypdf fallback."

        # 3. Fallback to pypdf if PyMuPDF failed or is not available
        if (not result["success"] or not result["text"]) and PYPDF_AVAILABLE:
            try:
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                result["page_count"] = len(reader.pages)
                pages_text = []
                for page in reader.pages:
                    p_text = page.extract_text() or ""
                    pages_text.append(p_text)

                result["pages"] = pages_text
                result["raw_text"] = "\n\n".join(pages_text).strip()
                result["text"] = result["raw_text"]
                result["parser_used"] = "pypdf"
                result["success"] = True
                result["error"] = None
            except Exception as e:
                result["error"] = f"Corrupted PDF document or unable to extract text: {str(e)}"
                return result

        if not result["success"]:
            if not result["error"]:
                result["error"] = "No PDF parser engine available or file could not be read."
            return result

        # 4. Compute metrics and validate content
        total_text = result["text"]
        words = total_text.split()
        result["word_count"] = len(words)
        result["char_count"] = len(total_text)

        # Scanned PDF Detection
        if result["page_count"] > 0 and result["word_count"] < self.min_words_threshold:
            result["is_scanned"] = True
            result["warning"] = (
                "Unable to extract readable text from this PDF. "
                "This document appears to be a scanned or image-only PDF. "
                "Please upload a standard text-based PDF or run OCR beforehand."
            )

        return result


# Global instance
pdf_parser = PDFParser()
