"""Run LDA topic modeling on CFPB complaint narratives."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "complaints_clean.csv"
OUTPUT_PATH = ROOT / "visuals" / "lda_topics.txt"


def run_topic_modeling(n_topics: int = 8, n_words: int = 10) -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=["clean_text"])

    vectorizer = CountVectorizer(max_features=10_000, stop_words="english", min_df=5)
    doc_term_matrix = vectorizer.fit_transform(df["clean_text"])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    lines = []
    for topic_idx, topic in enumerate(lda.components_, start=1):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [words[i] for i in top_indices]
        lines.append(f"Topic {topic_idx}: " + ", ".join(top_words))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved topics to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_topic_modeling()
