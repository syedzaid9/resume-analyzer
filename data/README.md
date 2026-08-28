# Dataset Documentation & Setup Guide — ResumeAI

## Overview
ResumeAI uses a supervised Machine Learning & Natural Language Processing pipeline to classify resumes into job domains and conduct deep gap analysis against target occupational requirements.

---

## 1. Expected Dataset Format

The primary raw dataset should be located at:
`data/raw/resume_dataset.csv`

### Required Columns:
* `Resume_ID` (or `ID`): Unique integer/string identifier for the resume.
* `Category` (or `Domain`): Target professional domain / job category (e.g., `Data Science`, `Java Developer`, `DevOps`, `Finance`, `HR`).
* `Resume_str` (or `Resume_Text` / `Resume`): Clean or raw full-text representation of the candidate resume.
* *(Optional)* `Resume_html`: Raw HTML markup if extracted from web profiles.

---

## 2. Recommended Public Dataset Sources

1. **Kaggle Resume Dataset (2,484+ Resumes across 24+ Categories)**
   - **Source:** Kaggle (Sneha Bilagi / Updated Resume Dataset)
   - **Link:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehabilagi/resume-dataset)
   - **Categories:** Information Technology, Data Science, Engineering, Finance, HR, Healthcare, Sales, Marketing/Digital Media, Design, Education, Business Development, Accounting, Banking, Cyber/IT, etc.

2. **Automated Setup / Fallback Generator:**
   - Run: `python scripts/setup_dataset.py`
   - If `data/raw/resume_dataset.csv` is not yet placed manually, this script verifies or generates a rich realistic dataset across all 24 categories so the ML pipeline and Streamlit dashboard function immediately without breaking.

---

## 3. Occupational & Domain Knowledge Base (`data/domain_requirements/`)

Skill requirements and domain mapping are derived from industry standards (O*NET 28.0 taxonomy and technical job descriptions):

* `occupations.csv`: 22 target career domains with role descriptions, education expectations, and experience standards.
* `skills.csv`: 120+ verified skills categorized by domain with extensive synonym / abbreviation aliases (e.g., `ML` -> `Machine Learning`, `JS` -> `JavaScript`, `PostgreSQL` -> `PostgreSQL / SQL`).
* `technologies.csv`: Catalog of software stacks, ecosystems, and platforms.
* `domain_mapping.csv`: Skill-to-domain relationships tagged by importance level (`Critical`, `High`, `Medium`, `Low`).

---

## 4. Execution Workflow

To preprocess the raw dataset and train the ML models:

```bash
# 1. Setup and inspect dataset
python scripts/setup_dataset.py

# 2. Preprocess text (preserves C++, .NET, React.js, etc.)
python scripts/preprocess_dataset.py

# 3. Train & compare 4 ML models (Logistic Regression, Naive Bayes, Linear SVM, Random Forest)
python scripts/train_model.py

# 4. Evaluate best model on test set & output genuine metrics
python scripts/evaluate_model.py
```
