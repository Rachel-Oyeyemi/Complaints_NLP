"""Download and sample CFPB consumer complaints.

This script downloads the public CFPB Consumer Complaint Database and keeps
complaints with consumer narratives. By default, it saves a laptop-friendly
sample for portfolio use.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests

CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_csv(zip_path: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        archive.extract(csv_name, output_dir)
        return output_dir / csv_name


def build_sample(sample_size: int, start_date: str | None, end_date: str | None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / "complaints.csv.zip"
    if not zip_path.exists():
        print("Downloading CFPB complaint data...")
        download_file(CFPB_URL, zip_path)

    csv_path = extract_csv(zip_path, RAW_DIR)
    output_path = PROCESSED_DIR / "complaints_sample.csv"

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
        chunk = chunk.dropna(subset=["Consumer complaint narrative", "Product"])
        if start_date:
            chunk = chunk[chunk["Date received"] >= start_date]
        if end_date:
            chunk = chunk[chunk["Date received"] <= end_date]
        if not chunk.empty:
            chunks.append(chunk)
        if sum(len(c) for c in chunks) >= sample_size:
            break

    if not chunks:
        raise ValueError("No records found. Try a different date range.")

    sample = pd.concat(chunks, ignore_index=True).head(sample_size)
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} complaints to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD start date")
    parser.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD end date")
    args = parser.parse_args()
    build_sample(args.sample_size, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
