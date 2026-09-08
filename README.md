# ResumeAI — AI-Powered Resume Analysis & Career Matching System

ResumeAI is a web application built with Streamlit that analyzes PDF resumes to help job seekers understand how well their resume aligns with target roles and modern Applicant Tracking Systems (ATS).

I built this project to explore how resume screening pipelines work in practice and to create an explainable alternative to black-box resume checkers. Instead of just querying an external LLM API, the app runs a full local machine learning and NLP pipeline to parse resumes, predict career tracks, match skills, and generate actionable feedback.

---

## What It Does

- **Resume Parsing:** Reads text from uploaded PDF resumes while handling common technical terms and abbreviations (like `C++`, `.NET`, `Node.js`, and `CI/CD`).
- **Career Domain Classification:** Analyzes the resume content and predicts the best-fitting job domain across 22 career tracks (such as Data Science, Web Development, DevOps, and Cyber Security), along with top alternative matches.
- **Skill Extraction & Gap Analysis:** Identifies technical skills and normalizes synonyms (e.g., mapping `ML` to `Machine Learning`). It compares detected skills against domain requirements to highlight what you have and what you are missing.
- **ATS Compatibility Audit:** Checks for standard resume sections, contact information, action verbs, and quantified achievements to see how parsable the resume is.
- **Job Description Matcher:** Lets you paste a specific job description to compare keyword alignment and see role-specific suggestions.
- **Scoring & Reports:** Generates an overall 0–100 fit score based on a transparent breakdown (skills, experience, projects, education, ATS check, structure, and formatting) and allows downloading the analysis as a PDF or Markdown report.

---

## How It Works

1. **Text Extraction:** Uses PyMuPDF (with a `pypdf` fallback) to extract raw text from PDF files and cleans it without losing programming tokens.
2. **ML Classification:** Converts resume text into numerical features using TF-IDF (unigrams and bigrams) and passes them to a supervised classifier (Logistic Regression) trained on multi-domain resume data to predict the target career domain.
3. **Rule-Based & Semantic Matching:** Matches extracted tokens against a categorized skills dataset, calculates cosine similarity between the resume and domain profiles, and audits standard ATS formatting rules.
4. **Scoring & Feedback:** Aggregates weighted scores across seven components to produce the final fit score, personalized improvement tips, and portfolio project suggestions.

---

## Tech Stack

- **Backend & ML:** Python, Scikit-learn, Pandas, NumPy, NLTK, Joblib
- **Document Processing:** PyMuPDF, pypdf, fpdf2
- **Frontend & Visualizations:** Streamlit, Plotly
- **Testing:** Pytest

---

## Getting Started

### 1. Set up a virtual environment
```bash
cd resume-analyzer
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to use the app.

### 4. (Optional) Run tests
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
resume-analyzer/
├── app.py                  # Main Streamlit dashboard
├── src/                    # Parsing, classification, scoring, and report generation modules
├── data/                   # Raw resume dataset and domain/skill taxonomies
├── models/                 # Saved classifier and TF-IDF vectorizer artifacts
├── scripts/                # Scripts to preprocess data, train, and evaluate the model
└── tests/                  # Unit and integration tests
```

---

## Future Improvements

- Add support for transformer-based embeddings (like Sentence-BERT) for deeper semantic comparisons.
- Support multi-page layout detection and two-column resume parsing.
- Expand the dataset to cover more non-technical career fields.
