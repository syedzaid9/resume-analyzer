# ResumeAI — AI-Powered Resume Analysis & Career Matching System

A local web application that analyzes PDF resumes, predicts career domains using supervised machine learning, audits ATS compatibility, and highlights skill gaps for target roles.

I built this project to understand how recruitment screening pipelines work under the hood and to build an explainable alternative to black-box resume checkers. Instead of wrapping an LLM API, ResumeAI uses a full machine learning and NLP pipeline—from custom text extraction to multiclass classification and weighted scoring.

---

## What It Does

- **PDF Parsing & Cleaning:** Extracts text from PDF resumes using PyMuPDF (with a `pypdf` fallback) and normalizes text while preserving technical tokens like `C++`, `C#`, `.NET`, `Node.js`, and `CI/CD`.
- **Domain Classification:** Classifies resumes across 22 career tracks (e.g., Data Science, DevOps, Full Stack, Cyber Security) and outputs the top 3 matching paths with confidence scores.
- **Skill Extraction & Gap Analysis:** Matches skills against an occupational knowledge base (300+ skills) with synonym handling (`JS` → `JavaScript`, `ML` → `Machine Learning`), prioritizing missing skills into Critical, High, Medium, and Low.
- **ATS Compatibility Audit:** Runs a 10-point structural check covering contact details, standard section headings, action verbs, measurable metrics, and parsability.
- **Job Description Matcher:** Compares resume content directly against pasted job descriptions to measure keyword alignment.
- **Explainable Scoring:** Calculates an overall 0–100 fit score across 7 transparent components.
- **Exportable Reports:** Generates downloadable summary reports in PDF (via `fpdf2`) and Markdown formats.
- **Model Diagnostics:** Includes a built-in diagnostics tab showing confusion matrices and evaluation metrics.

---

## Machine Learning Pipeline & Results

The domain classification model was trained on a dataset of 264 resume samples across 22 professional domains.

1. **Preprocessing & Feature Extraction:** Text is normalized and vectorized using TF-IDF (unigrams + bigrams, sublinear term-frequency scaling, ~2,833 vocabulary size).
2. **Model Selection:** I evaluated four supervised classifiers using stratified splits (60% train, 20% validation, 20% held-out test):

| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **100%** | **1.00** | **98.11%** | **0.98** |
| Calibrated Linear SVM | 100% | 1.00 | 98.11% | 0.98 |
| Multinomial Naive Bayes | 100% | 1.00 | 96.23% | 0.95 |
| Random Forest | 94.34% | 0.94 | 92.45% | 0.91 |

Logistic Regression was chosen for inference due to its strong generalization on the test set, fast inference time (~0.07s), and well-calibrated class probabilities.

---

## Scoring System

Instead of a single arbitrary number, the overall fit score (0–100) is calculated from 7 weighted components defined in `src/config.py`:

- **Skill Match (30%):** Weighted coverage of required domain skills.
- **Experience (20%):** Relevant job titles, action verbs, and quantified impact.
- **Projects (15%):** Depth of technical portfolio and domain relevance.
- **Education (10%):** Degree level and field of study alignment.
- **ATS Compatibility (10%):** File parsability, clean headers, and contact details.
- **Structure (10%):** Section completeness (Summary, Skills, Experience, Education, Projects).
- **Formatting (5%):** Word count range and readability.

---

## Tech Stack

- **Core / ML:** Python, Scikit-learn, NumPy, Pandas, NLTK, Joblib
- **Document Processing:** PyMuPDF (`fitz`), `pypdf`, `fpdf2`
- **UI & Visualization:** Streamlit, Plotly
- **Testing:** Pytest (12 unit and integration tests)

---

## Project Structure

```
resume-analyzer/
├── app.py                     # Streamlit dashboard and UI logic
├── src/                       # Core modules (parsers, classifiers, scorers)
│   ├── pdf_parser.py          # Dual PyMuPDF/pypdf text extractor
│   ├── text_preprocessor.py   # Token-preserving text cleaner
│   ├── skill_extractor.py     # Skill matcher and synonym normalizer
│   ├── domain_classifier.py   # Model inference and probability ranking
│   ├── ats_analyzer.py        # 10-point ATS compliance rule engine
│   ├── scoring_engine.py      # 7-component scoring formula
│   └── report_generator.py    # PDF and Markdown export
├── data/                      # Raw/processed data and domain taxonomies
├── models/                    # Serialized models, vectorizer, and metadata
├── scripts/                   # Dataset setup, training, and evaluation scripts
└── tests/                     # Pytest suite
```

---

## Getting Started

### 1. Setup Environment
```bash
# Clone and enter directory
cd resume-analyzer

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Train / Evaluate Models
Pre-trained model artifacts are included in `models/`, but you can re-run the pipeline from scratch:
```bash
python scripts/preprocess_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

### 3. Run the App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Tests
```bash
pytest tests/ -v
```

---

## Future Improvements

- Add dense semantic embeddings (e.g., Sentence Transformers / `all-MiniLM-L6-v2`) to complement TF-IDF matching.
- Expand the dataset to include more non-tech industry domains.
- Add support for multi-page layout and two-column resume parsing detection.
