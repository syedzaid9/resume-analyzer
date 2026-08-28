"""Data Preprocessing Script for ResumeAI.
Loads raw resume dataset, performs text cleaning while preserving technical tokens,
removes duplicates and null records, and saves the cleaned dataset to data/processed/cleaned_resumes.csv.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH, PROCESSED_DATA_DIR
from src.text_preprocessor import TextPreprocessor


def identify_columns(df: pd.DataFrame):
    """Accurately identifies category, text, and ID columns without collision."""
    cols = list(df.columns)
    
    # 1. ID column
    id_col = next((c for c in cols if c.strip().lower() in ["resume_id", "id", "candidate_id"]), None)
    
    # 2. Category / Domain column
    cat_col = next((c for c in cols if c.strip().lower() in ["category", "domain", "role", "job_title", "label", "target"]), None)
    
    # 3. Text column (prioritize resume_str, resume_text, text, content, resume)
    text_candidates = ["resume_str", "resume_text", "cleaned_resume", "resume", "text", "content", "skills_and_experience"]
    text_col = next((c for c in cols if c.strip().lower() in text_candidates), None)
    
    if not text_col:
        # Fallback: look for longest average string column that is not category or id
        str_cols = [c for c in cols if c != cat_col and c != id_col and df[c].dtype == object]
        if str_cols:
            text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())

    return id_col, cat_col, text_col


def preprocess_dataset() -> pd.DataFrame:
    """Loads raw dataset, cleans text, and saves to processed path."""
    if not RAW_DATASET_PATH.exists():
        print(f"[ERROR] Raw dataset not found at {RAW_DATASET_PATH}. Run scripts/setup_dataset.py first.")
        sys.exit(1)

    print(f"[INFO] Loading raw dataset from: {RAW_DATASET_PATH}")
    df_raw = pd.read_csv(RAW_DATASET_PATH)

    initial_count = len(df_raw)
    print(f"[INFO] Initial record count: {initial_count}")

    id_col, cat_col, text_col = identify_columns(df_raw)
    print(f"[INFO] Detected columns -> ID: '{id_col}', Category: '{cat_col}', Text: '{text_col}'")

    if not cat_col or not text_col:
        print("[ERROR] Required columns ('Category' and 'Resume_str') could not be identified.")
        sys.exit(1)

    # Build clean standardized DataFrame
    df = pd.DataFrame()
    df["Resume_ID"] = df_raw[id_col].astype(str) if id_col else [str(i) for i in range(1, len(df_raw) + 1)]
    df["Category"] = df_raw[cat_col].astype(str).str.strip()
    df["Resume_str"] = df_raw[text_col].astype(str)

    # Reset index to guarantee unique index
    df = df.reset_index(drop=True)

    # Drop nulls
    df = df.dropna(subset=["Category", "Resume_str"]).copy()

    # Drop exact duplicates in text
    df = df.drop_duplicates(subset=["Resume_str"]).reset_index(drop=True)

    # Clean text using domain-aware preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=False)
    print("[INFO] Cleaning and normalizing resume text (preserving technical terms: C++, .NET, React.js, etc.)...")
    
    cleaned_texts = [preprocessor.clean_text_for_ml(txt) for txt in df["Resume_str"]]
    df["Cleaned_Resume"] = cleaned_texts

    # Compute text length metrics
    df["Word_Count"] = [len(t.split()) for t in df["Cleaned_Resume"]]
    df["Char_Count"] = [len(t) for t in df["Cleaned_Resume"]]

    # Filter out empty or trivially short records
    df = df[df["Word_Count"] >= 10].reset_index(drop=True)

    # Save to processed directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Raw Resumes Processed: {initial_count}")
    print(f"Cleaned Resumes Retained: {len(df)}")
    print(f"Unique Categories: {df['Category'].nunique()}")
    print(f"Average Word Count per Resume: {df['Word_Count'].mean():.1f} words")
    print(f"Min Word Count: {df['Word_Count'].min()} | Max Word Count: {df['Word_Count'].max()}")
    print(f"Processed Dataset Saved To: {PROCESSED_DATASET_PATH}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    preprocess_dataset()
