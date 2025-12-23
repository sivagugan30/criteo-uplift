# Criteo Uplift Dataset

This repository contains the [Criteo Uplift dataset](https://huggingface.co/datasets/criteo/criteo-uplift), a large-scale benchmark dataset for uplift modeling.

## Dataset Description

The Criteo Uplift dataset is designed for research in uplift modeling (also known as causal inference in marketing). It contains approximately 25 million rows of anonymized data from a randomized controlled trial.

### Features

- **f0-f11**: 12 anonymized features (float values)
- **treatment**: Binary indicator (0 = control, 1 = treatment)
- **conversion**: Binary outcome variable
- **visit**: Binary indicator for website visit
- **exposure**: Binary indicator for ad exposure

## Project Structure

```
criteo-uplift/
├── data/
│   ├── raw/                    # Original dataset files
│   │   └── criteo_uplift.parquet
│   ├── processed/              # Processed/sample files
│   │   └── criteo_uplift_sample.csv
│   └── dataset_info.txt        # Dataset metadata
├── scripts/
│   └── download_dataset.py     # Script to download the dataset
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset:
   ```bash
   python scripts/download_dataset.py
   ```

## Usage

### Loading the full dataset (Parquet)

```python
import pandas as pd

df = pd.read_parquet("data/raw/criteo_uplift.parquet")
print(df.shape)  # (~25M rows, 16 columns)
```

### Loading from Hugging Face directly

```python
from datasets import load_dataset

ds = load_dataset("criteo/criteo-uplift")
```

## License

The dataset is provided by Criteo and is subject to their terms of use. See the [Hugging Face dataset page](https://huggingface.co/datasets/criteo/criteo-uplift) for more details.

