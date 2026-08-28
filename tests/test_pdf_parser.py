"""Integration test for sample PDF parsing and full analysis flow."""

from pathlib import Path
import pytest
from src.pdf_parser import pdf_parser
from src.resume_parser import resume_parser
from src.skill_extractor import skill_extractor
from src.domain_classifier import domain_classifier
from src.scoring_engine import scoring_engine
from src.ats_analyzer import ats_analyzer
from src.report_generator import report_generator

SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_resume.pdf"


def test_sample_pdf_pipeline():
    if not SAMPLE_PDF_PATH.exists():
        pytest.skip("sample_resume.pdf not found")

    # 1. Parse PDF
    parse_res = pdf_parser.parse_pdf(SAMPLE_PDF_PATH)
    assert parse_res["success"] is True
    assert parse_res["word_count"] > 50
    assert "Alex Mercer" in parse_res["text"]

    # 2. Extract Info & Skills
    parsed_info = resume_parser.parse_full_resume(parse_res["text"])
    skills_data = skill_extractor.extract_skills(parse_res["text"])
    skills = skills_data["all_detected_skills"]
    assert "Python" in skills
    assert "PyTorch" in skills
    assert "Docker" in skills

    # 3. Domain classification
    pred = domain_classifier.predict_domain(parse_res["text"])
    assert pred["success"] is True
    assert pred["primary_domain"] in ["Data Science", "Machine Learning"]

    # 4. ATS & Scoring
    ats_res = ats_analyzer.analyze(parse_res["text"], parsed_info, skills, target_domain="Data Science")
    assert ats_res["ats_score"] >= 75

    score_res = scoring_engine.compute_overall_score(
        parse_res["text"], parsed_info, skills, ats_res, target_domain="Data Science"
    )
    assert score_res["overall_fit_score"] >= 70

    # 5. Report Generation
    full_analysis = {
        "target_domain": "Data Science",
        "overall_fit_score": score_res["overall_fit_score"],
        "rating_label": score_res["rating_label"],
        "domain_match_pct": 88.5,
        "parsed_data": parsed_info,
        "scoring_data": score_res,
        "ats_data": ats_res,
        "strengths": ["Strong Python skills"],
        "weaknesses": ["Minor gaps"],
        "improvements": [{"category": "Action Verbs", "recommendation": "Use Google XYZ"}],
    }
    pdf_out = report_generator.generate_pdf_report(full_analysis)
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 0
