"""
Train and save the Saudi Tech Role Predictor model.

This script is designed as an extension to the existing project:
Saudi Tech Job Skills Analysis.

It does not modify the original notebook or dataset.
It creates:
- role_model.pkl
- tfidf_vectorizer.pkl

Run from the app folder:
    python train_and_save_model.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = APP_DIR / "role_model.pkl"
VECTORIZER_PATH = APP_DIR / "tfidf_vectorizer.pkl"

TARGET_ROLES = [
    "ai/ml engineer",
    "data analyst",
    "data engineer",
    "data scientist",
]


def find_dataset() -> Path:
    """Find the best dataset available in the project."""
    candidates = [
        PROJECT_ROOT / "data" / "jobs_sa_model_ready.csv",
        PROJECT_ROOT / "jobs_sa_model_ready.csv",
        PROJECT_ROOT / "data" / "jobs_sa_cleaned.csv",
        PROJECT_ROOT / "jobs_sa_cleaned.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"✅ Dataset found: {path}")
            return path

    checked_paths = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the dataset. Checked these paths:\n"
        f"{checked_paths}\n\n"
        "Make sure you placed the app folder inside the project root."
    )


def read_dataset(path: Path) -> pd.DataFrame:
    """Read CSV robustly using common encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"✅ Loaded dataset using encoding: {encoding}")
            print(f"Dataset shape: {df.shape}")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read dataset: {path}\nLast error: {last_error}")


def clean_job_title(title: str) -> str:
    """Standardize job titles the same way used in the modelling notebook."""
    title = str(title).lower()

    remove_words = [
        "senior", "junior", "lead", "principal", "intern",
        "remote", "mid-level", "mid level", "entry level",
        "manager", "specialist", "sr", "jr",
    ]

    for word in remove_words:
        title = title.replace(word, "")

    title = re.sub(r"[^a-zA-Z\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def map_role(title: str) -> str:
    """Map detailed job titles into the four role categories."""
    title = str(title).lower()

    if "data scientist" in title or "data science" in title:
        return "data scientist"
    if "data analyst" in title or "analytics" in title:
        return "data analyst"
    if "data engineer" in title:
        return "data engineer"
    if "ai" in title or "machine learning" in title or "ml" in title:
        return "ai/ml engineer"

    return "other"


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect feature and label columns.

    Preferred notebook-ready columns:
    - feature: skills_text
    - label: role_label

    Fallback notebook source columns:
    - feature: clean_description
    - label: job_title, then role_label is reconstructed
    """
    columns = set(df.columns)

    if {"skills_text", "role_label"}.issubset(columns):
        return "skills_text", "role_label"

    if {"clean_description", "role_label"}.issubset(columns):
        return "clean_description", "role_label"

    if {"clean_description", "job_title"}.issubset(columns):
        df["clean_title"] = df["job_title"].apply(clean_job_title)
        df["role_label"] = df["clean_title"].apply(map_role)
        return "clean_description", "role_label"

    if {"job_description", "job_title"}.issubset(columns):
        df["clean_title"] = df["job_title"].apply(clean_job_title)
        df["role_label"] = df["clean_title"].apply(map_role)
        return "job_description", "role_label"

    raise ValueError(
        "Could not detect the required columns.\n"
        "Expected one of these combinations:\n"
        "- skills_text + role_label\n"
        "- clean_description + role_label\n"
        "- clean_description + job_title\n"
        "- job_description + job_title\n\n"
        f"Available columns are:\n{list(df.columns)}"
    )


def prepare_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Prepare X text and y labels for training."""
    feature_col, label_col = detect_columns(df)
    print(f"✅ Feature column used as X: {feature_col}")
    print(f"✅ Label column used as y: {label_col}")

    model_df = df[[feature_col, label_col]].copy()
    model_df = model_df.rename(columns={feature_col: "skills_text", label_col: "role_label"})

    model_df["skills_text"] = model_df["skills_text"].fillna("").astype(str).str.strip()
    model_df["role_label"] = model_df["role_label"].fillna("").astype(str).str.lower().str.strip()

    model_df = model_df[model_df["skills_text"] != ""]
    model_df = model_df[model_df["role_label"].isin(TARGET_ROLES)].copy()

    if model_df.empty:
        raise ValueError(
            "No rows remain after filtering to the four target roles. "
            "Check that role_label/job_title values match the expected roles."
        )

    print("\nClass distribution after filtering:")
    print(model_df["role_label"].value_counts())

    min_class_count = model_df["role_label"].value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples, so stratified train/test split is not possible.\n"
            f"Class counts:\n{model_df['role_label'].value_counts()}"
        )

    return model_df["skills_text"], model_df["role_label"]


def main() -> None:
    dataset_path = find_dataset()
    df = read_dataset(dataset_path)
    X_text, y = prepare_data(df)

    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X = tfidf.fit_transform(X_text)

    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set: X_test={X_test.shape}, y_test={y_test.shape}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\nRandom Forest Evaluation")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    print("\n✅ Model saved successfully:")
    print(f"   {MODEL_PATH}")
    print("✅ TF-IDF vectorizer saved successfully:")
    print(f"   {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
