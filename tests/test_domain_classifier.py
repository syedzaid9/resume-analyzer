"""Unit tests for DomainClassifier, ATSAnalyzer, ScoringEngine, JobMatcher, and PDFParser."""

import pytest
from src.domain_classifier import DomainClassifier
from src.ats_analyzer import ATSAnalyzer
from src.scoring_engine import ScoringEngine
from src.job_matcher import JobMatcher
from src.pdf_parser import PDFParser
from src.resume_parser import ResumeParser


def test_domain_classifier_inference():
    classifier = DomainClassifier()
    assert classifier.is_model_ready() is True

    ds_text = (
        "Senior Data Scientist with 5 years experience in machine learning, Python, SQL, "
        "Pandas, Scikit-learn, and predictive modeling."
    )
    result = classifier.predict_domain(ds_text, top_k=3)
    assert result["success"] is True
    assert result["primary_domain"] in ["Data Science", "Machine Learning"]
    assert len(result["top_domains"]) == 3
    assert result["top_domains"][0]["confidence_pct"] > 0


def test_ats_analyzer():
    analyzer = ATSAnalyzer()
    parser = ResumeParser()
    sample_resume = """
    John Doe
    john.doe@email.com | 123-456-7890 | linkedin.com/in/johndoe
    
    Professional Summary
    Senior Software Engineer with 5 years experience building scalable backend APIs.
    
    Technical Skills
    Python, FastAPI, Docker, PostgreSQL, AWS, Git
    
    Work Experience
    Senior Developer at TechCorp
    - Engineered microservices architecture using Python and FastAPI, increasing API throughput by 35%.
    - Automated CI/CD deployment pipelines on AWS using Docker.
    
    Education
    Bachelor of Science in Computer Science, State University, 2020
    """
    parsed = parser.parse_full_resume(sample_resume)
    skills = ["Python", "FastAPI", "Docker", "PostgreSQL", "AWS", "Git"]
    ats_res = analyzer.analyze(sample_resume, parsed, skills, target_domain="Python Developer")

    assert ats_res["ats_score"] >= 70
    assert len(ats_res["strengths"]) > 0


def test_scoring_engine_weights():
    engine = ScoringEngine()
    parser = ResumeParser()
    analyzer = ATSAnalyzer()
    sample_text = """
    Jane Smith
    jane@email.com | 555-123-4567 | linkedin.com/in/janesmith
    
    Summary: Lead Data Scientist specializing in Machine Learning.
    Skills: Python, SQL, Pandas, NumPy, Scikit-learn, Docker, Machine Learning, Deep Learning
    Experience: 4 years designing neural network architectures and predictive models.
    Education: Master of Science in Data Science, 2021.
    """
    parsed = parser.parse_full_resume(sample_text)
    skills = ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Docker", "Machine Learning", "Deep Learning"]
    ats_res = analyzer.analyze(sample_text, parsed, skills, target_domain="Data Science")

    score_data = engine.compute_overall_score(
        sample_text, parsed, skills, ats_res, target_domain="Data Science"
    )

    assert 0 <= score_data["overall_fit_score"] <= 100
    assert score_data["rating_label"] in ["Excellent", "Very Good", "Good", "Needs Improvement"]
    # Check component breakdown sum matches
    breakdown = score_data["breakdown"]
    total_earned = sum(item["earned"] for item in breakdown.values())
    assert abs(total_earned - score_data["overall_fit_score"]) <= 1.0


def test_job_matcher():
    matcher = JobMatcher()
    resume_text = "Experienced in Python, SQL, Docker, FastAPI, and Git."
    jd_text = """
    We are seeking a Senior Python Developer.
    Requirements:
    - Proficiency in Python, FastAPI, and SQL.
    - Experience with Docker and AWS.
    - Strong problem solving and communication skills.
    """
    res = matcher.match_job_description(resume_text, jd_text, ["Python", "SQL", "Docker", "FastAPI", "Git"])

    assert res["success"] is True
    assert res["match_pct"] > 40
    assert "Python" in res["matched_skills"]
    assert "FastAPI" in res["matched_skills"]


def test_pdf_parser_empty():
    parser = PDFParser()
    res = parser.parse_pdf(b"")
    assert res["success"] is False
    assert "empty" in res["error"].lower() or "no data" in res["error"].lower()
