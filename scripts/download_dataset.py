#!/usr/bin/env python3
"""
Download and save the Criteo Uplift dataset in a structured format.

The Criteo Uplift dataset is used for uplift modeling research.
It contains ~25M rows with features, treatment indicator, and conversion labels.
"""

import os
from pathlib import Path

from datasets import load_dataset


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Criteo Uplift dataset from Hugging Face...")
    ds = load_dataset("criteo/criteo-uplift")

    print(f"Dataset structure: {ds}")
    print(f"Number of examples: {ds['train'].num_rows:,}")

    # Save as Parquet (efficient columnar format)
    parquet_path = raw_dir / "criteo_uplift.parquet"
    print(f"\nSaving dataset to {parquet_path}...")
    ds["train"].to_parquet(str(parquet_path))
    print(f"Saved Parquet file: {parquet_path}")

    # Also save as CSV for easy inspection (first 10k rows as sample)
    csv_sample_path = processed_dir / "criteo_uplift_sample.csv"
    print(f"\nSaving sample (first 10,000 rows) to {csv_sample_path}...")
    ds["train"].select(range(10000)).to_csv(str(csv_sample_path))
    print(f"Saved CSV sample: {csv_sample_path}")

    # Print dataset info
    print("\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print(f"Features: {ds['train'].features}")
    print(f"\nColumn names: {ds['train'].column_names}")
    print(f"Total rows: {ds['train'].num_rows:,}")

    # Save dataset info to a text file
    info_path = data_dir / "dataset_info.txt"
    with open(info_path, "w") as f:
        f.write("Criteo Uplift Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: https://huggingface.co/datasets/criteo/criteo-uplift\n\n")
        f.write(f"Total rows: {ds['train'].num_rows:,}\n\n")
        f.write("Features:\n")
        for name, feature in ds["train"].features.items():
            f.write(f"  - {name}: {feature}\n")
        f.write("\n")
        f.write("Files:\n")
        f.write(f"  - raw/criteo_uplift.parquet: Full dataset in Parquet format\n")
        f.write(f"  - processed/criteo_uplift_sample.csv: Sample of first 10,000 rows\n")

    print(f"\nDataset info saved to: {info_path}")
    print("\nDownload complete!")


if __name__ == "__main__":
    main()

