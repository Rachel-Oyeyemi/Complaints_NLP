"""Clean CFPB complaint narratives for NLP modeling."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "complaints_sample.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "complaints_clean.csv"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = df.dropna(subset=["Consumer complaint narrative", "Product"]).copy()
    df["clean_text"] = df["Consumer complaint narrative"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 20]

    keep_columns = [
        "Date received",
        "Product",
        "Issue",
        "Company",
        "State",
        "Consumer complaint narrative",
        "clean_text",
    ]
    available_columns = [col for col in keep_columns if col in df.columns]
    df = df[available_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset with {len(df):,} rows to {output_path}")
    return df


if __name__ == "__main__":
    preprocess()
