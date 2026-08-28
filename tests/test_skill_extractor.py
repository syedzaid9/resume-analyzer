"""Unit tests for SkillExtractor."""

import pytest
from src.skill_extractor import SkillExtractor


@pytest.fixture
def extractor():
    return SkillExtractor()


def test_extract_canonical_skills(extractor):
    text = "Developed predictive models using Python, Scikit-learn, Pandas, and SQL with Docker on AWS."
    res = extractor.extract_skills(text)
    skills = res["all_detected_skills"]
    
    assert "Python" in skills
    assert "Scikit-learn" in skills
    assert "Pandas" in skills
    assert "SQL" in skills
    assert "Docker" in skills
    assert "AWS" in skills


def test_extract_synonyms(extractor):
    text = "Experienced with ReactJS, NodeJS, K8s, and ML algorithms."
    res = extractor.extract_skills(text)
    skills = res["all_detected_skills"]
    
    assert "React" in skills
    assert "Node.js" in skills
    assert "Kubernetes" in skills
    assert "Machine Learning" in skills


def test_categories_assigned(extractor):
    text = "Full Stack developer proficient in JavaScript, PostgreSQL, Docker, and PyTorch."
    res = extractor.extract_skills(text)
    cats = res["categorized_skills"]
    
    assert "JavaScript" in cats["Programming Languages"]
    assert "PostgreSQL" in cats["Databases"]
    assert "Docker" in cats["Cloud & DevOps"]
    assert "PyTorch" in cats["AI / Machine Learning"]
