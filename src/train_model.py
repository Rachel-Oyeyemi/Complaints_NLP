"""Train a TF-IDF + Logistic Regression complaint classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "complaints_clean.csv"
MODEL_PATH = ROOT / "models" / "tfidf_logistic_model.joblib"
REPORT_PATH = ROOT / "visuals" / "classification_report.txt"


def train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> Pipeline:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["clean_text", "Product"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["Product"],
        test_size=0.2,
        random_state=42,
        stratify=df["Product"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    stop_words="english",
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print(report)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    REPORT_PATH.write_text(report + "\n\nConfusion Matrix:\n" + str(matrix), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved report to {REPORT_PATH}")
    return pipeline


if __name__ == "__main__":
    train_model()
