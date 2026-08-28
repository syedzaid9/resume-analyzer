"""Unit tests for TextPreprocessor."""

import pytest
from src.text_preprocessor import TextPreprocessor


def test_clean_html():
    p = TextPreprocessor()
    html_text = "<p>Experienced in <b>Python</b> and <i>Machine Learning</i>.</p>"
    clean = p.clean_html(html_text)
    assert "<p>" not in clean
    assert "<b>" not in clean
    assert "Python" in clean


def test_tech_tokens_preservation():
    p = TextPreprocessor()
    raw = "Skills: C++, C#, .NET Core, Node.js, React.js, CI/CD, and AWS."
    cleaned = p.clean_text_for_ml(raw)
    
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned
    assert "node.js" in cleaned
    assert "react.js" in cleaned
    assert "ci/cd" in cleaned
    assert "aws" in cleaned


def test_unicode_normalization():
    p = TextPreprocessor()
    unicode_text = "Senior Developer – 5 years’ experience • Built microservices"
    norm = p.normalize_unicode(unicode_text)
    assert "–" not in norm
    assert "-" in norm
    assert "'" in norm
