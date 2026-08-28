"""Generates standard, fully-formed Jupyter Notebooks (.ipynb) for EDA, Preprocessing, Training, and Evaluation.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# ==========================================
# Notebook 1: 01_data_exploration.ipynb
# ==========================================
nb1_cells = [
    md_cell("# 01. Exploratory Data Analysis (EDA) — ResumeAI\nThis notebook performs comprehensive exploratory data analysis on the resume classification dataset."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline"""),
    md_cell("## 1. Load Raw Dataset"),
    code_cell("""df = pd.read_csv('../data/raw/resume_dataset.csv')
print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
df.head()"""),
    md_cell("## 2. Check Missing Values and Duplicates"),
    code_cell("""print('--- Missing Values ---')
print(df.isnull().sum())

print('\\n--- Duplicates ---')
duplicates = df.duplicated(subset=['Resume_str']).sum()
print(f'Duplicate resume texts: {duplicates}')"""),
    md_cell("## 3. Category Distribution Analysis"),
    code_cell("""category_counts = df['Category'].value_counts()
print(category_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Resume Category Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Number of Resumes')
plt.ylabel('Professional Domain')
plt.tight_layout()
plt.show()"""),
    md_cell("## 4. Resume Text Length & Word Count Distribution"),
    code_cell("""df['word_count'] = df['Resume_str'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['Resume_str'].apply(lambda x: len(str(x)))

print(df[['word_count', 'char_count']].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['word_count'], kde=True, ax=axes[0], color='#2563EB', bins=25)
axes[0].set_title('Word Count Distribution', fontweight='bold')

sns.boxplot(x='Category', y='word_count', data=df, ax=axes[1], palette='tab20')
axes[1].set_title('Word Count by Category', fontweight='bold')
axes[1].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()""")
]

# ==========================================
# Notebook 2: 02_preprocessing.ipynb
# ==========================================
nb2_cells = [
    md_cell("# 02. Data Preprocessing & NLP Normalization — ResumeAI\nCleans text while preserving critical technical tokens like C++, .NET, and React.js."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, '..')
from src.text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()"""),
    md_cell("## 1. Test Technical Token Preservation"),
    code_cell("""test_sample = 'Experienced in C++, C#, .NET Core, Node.js, React.js, CI/CD, and AWS cloud.'
cleaned_sample = preprocessor.clean_text_for_ml(test_sample)
print('Original:', test_sample)
print('Cleaned: ', cleaned_sample)"""),
    md_cell("## 2. Process Entire Raw Dataset"),
    code_cell("""raw_df = pd.read_csv('../data/raw/resume_dataset.csv')
raw_df['Cleaned_Resume'] = raw_df['Resume_str'].apply(lambda x: preprocessor.clean_text_for_ml(str(x)))
raw_df['Cleaned_Word_Count'] = raw_df['Cleaned_Resume'].apply(lambda x: len(x.split()))

print('Cleaned dataset preview:')
raw_df[['Category', 'Cleaned_Word_Count', 'Cleaned_Resume']].head()"""),
    md_cell("## 3. Save Processed Dataset"),
    code_cell("""raw_df.to_csv('../data/processed/cleaned_resumes.csv', index=False)
print('Processed dataset saved successfully!')""")
]

# ==========================================
# Notebook 3: 03_model_training.ipynb
# ==========================================
nb3_cells = [
    md_cell("# 03. Supervised Model Training & Model Comparison — ResumeAI\nVectorizes text with TF-IDF and trains Logistic Regression, Naive Bayes, Linear SVM, and Random Forest."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib"""),
    md_cell("## 1. Load Processed Data & Split"),
    code_cell("""df = pd.read_csv('../data/processed/cleaned_resumes.csv')
X = df['Cleaned_Resume'].astype(str).values
y = df['Category'].astype(str).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f'Train: {len(X_train)}, Validation: {len(X_val)}')"""),
    md_cell("## 2. TF-IDF Vectorization"),
    code_cell("""vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3500, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')"""),
    md_cell("## 3. Train and Compare Models"),
    code_cell("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Multinomial NB': MultinomialNB(alpha=0.1),
    'Linear SVM': CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_val_vec)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro', zero_division=0)
    results.append({'Model': name, 'Accuracy': acc, 'Macro F1': f1})

res_df = pd.DataFrame(results).sort_values(by='Macro F1', ascending=False)
res_df""")
]

# ==========================================
# Notebook 4: 04_model_evaluation.ipynb
# ==========================================
nb4_cells = [
    md_cell("# 04. Model Evaluation & Performance Analysis — ResumeAI\nEvaluates the final selected model on held-out test data and plots the confusion matrix."),
    code_cell("""import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score"""),
    md_cell("## 1. Load Model & Vectorizer"),
    code_cell("""model = joblib.load('../models/classifier/best_classifier.joblib')
vectorizer = joblib.load('../models/vectorizer/tfidf_vectorizer.joblib')

with open('../models/model_metadata.json', 'r') as f:
    meta = json.load(f)

print(f'Loaded Best Model: {meta.get(\"best_model_name\")}')
print(f'Test Accuracy Recorded: {meta.get(\"test_metrics\", {}).get(\"accuracy\", 0) * 100:.2f}%')"""),
    md_cell("## 2. Test Set Evaluation"),
    code_cell("""df = pd.read_csv('../data/processed/cleaned_resumes.csv')
X = df['Cleaned_Resume'].astype(str).values
y = df['Category'].astype(str).values

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred, zero_division=0))"""),
    md_cell("## 3. Confusion Matrix Plot"),
    code_cell("""cm = confusion_matrix(y_test, y_pred, labels=meta['classes'])
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=meta['classes'], yticklabels=meta['classes'])
plt.title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Domain')
plt.ylabel('Actual Domain')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()""")
]

# Write all 4 notebooks
notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_preprocessing.ipynb": nb2_cells,
    "03_model_training.ipynb": nb3_cells,
    "04_model_evaluation.ipynb": nb4_cells,
}

for filename, cells in notebooks.items():
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated notebook: {path}")
