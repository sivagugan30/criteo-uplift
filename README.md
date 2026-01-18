# Criteo Uplift Modeling

An end-to-end uplift modeling project using the [Criteo Uplift Dataset](https://huggingface.co/datasets/criteo/criteo-uplift). Includes exploratory analysis, model training (S-Learner, T-Learner, X-Learner), evaluation, and an interactive Streamlit dashboard.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://criteo-uplift.streamlit.app)

## 📊 What is Uplift Modeling?

Uplift modeling predicts the **incremental impact** of a treatment (like showing an ad) on an individual's behavior. It answers: *"Who will convert **because of** the ad, not just who will convert?"*

### The Four User Types

| Segment | Without Ad | With Ad | Action |
|---------|------------|---------|--------|
| **Persuadables** | No | Yes | ✅ Target these! |
| **Sure Things** | Yes | Yes | Save budget |
| **Lost Causes** | No | No | Don't waste resources |
| **Sleeping Dogs** | Yes | No | 🚫 Avoid! |

## 📁 Project Structure

```
criteo-uplift/
├── streamlit_app/
│   └── app_v2.py              # Main Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_uplift_modeling.ipynb   # S/T/X Learner training
│   ├── 03_causal_forest.ipynb # Causal Forest experiments
│   ├── 04_advanced_evaluation.ipynb  # Qini curves, AUUC
│   └── 05_customer_profiles.ipynb    # Segmentation & SHAP
├── visualizations/
│   ├── data/                  # Pre-computed CSVs for dashboard
│   └── images/                # Saved plots
├── models/                    # Trained model artifacts
├── data/
│   ├── raw/                   # Original parquet (gitignored)
│   └── processed/             # Sample CSV (gitignored)
├── scripts/
│   └── download_dataset.py    # Download from HuggingFace
├── requirements.txt           # Streamlit Cloud dependencies
└── README.md
```

## 🖥️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/criteo-uplift.git
cd criteo-uplift

# Create virtual environment
python -m venv criteo-env
source criteo-env/bin/activate  # Windows: criteo-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app_v2.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path**: `streamlit_app/app_v2.py`
5. Deploy!

## 📚 References

- **Dataset Paper**: [A Large Scale Benchmark for Uplift Modeling](https://bitlater.github.io/files/large-scale-benchmark_comAH.pdf) (Diemert et al., Criteo Research, 2018)
- **CausalML Library**: [Uber's CausalML](https://github.com/uber/causalml)

## 📈 Key Findings

- **T-Learner** performed best on this dataset (Qini Coefficient: 35.19)
- The curve flattens after ~20%. Beyond that, you're paying for diminishing returns.

## License

The dataset is provided by Criteo. See the [HuggingFace dataset page](https://huggingface.co/datasets/criteo/criteo-uplift) for terms of use.
