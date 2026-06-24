"""Load a trained model and classify a new complaint narrative."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_logistic_model.joblib"


def predict(text: str, model_path: Path = MODEL_PATH) -> str:
    model = joblib.load(model_path)
    return str(model.predict([text])[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Complaint narrative to classify")
    args = parser.parse_args()
    print(predict(args.text))


if __name__ == "__main__":
    main()
