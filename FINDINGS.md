# Uplift Modeling on Criteo Dataset - Project Summary

## TL;DR - Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | T-Learner |
| **Qini Coefficient** | 35.19 |
| **AUUC** | 0.339% |
| **Dataset Size** | ~14M samples |
| **ATE** | +0.115 percentage points (~59% relative lift) |

### Customer Segments (Rule-Based)

| Segment | % of Users | Action |
|---------|------------|--------|
| Persuadables | 25% | Target these |
| Sure Things | 22% | Save budget |
| Lost Causes | 27% | Don't waste resources |
| Sleeping Dogs | 25% | Avoid treating |

---

## Overview

This project implements and evaluates uplift modeling techniques on the **Criteo Uplift Dataset**, a large-scale benchmark dataset from a randomized controlled trial (RCT) in digital advertising.

---

## Dataset Summary

| Attribute | Value |
|-----------|-------|
| **Source** | [Criteo/Hugging Face](https://huggingface.co/datasets/criteo/criteo-uplift) |
| **Total Samples** | 13,979,592 |
| **Features** | 12 anonymized features (f0-f11) |
| **Treatment Split** | 85% Treatment / 15% Control |
| **Conversion Rate** | ~0.29% (heavily imbalanced) |

### Why 85% Treatment?

Unlike typical product A/B tests (where treatment is small to minimize risk), this is an **advertising dataset**:
- Treatment = Showing an ad = Making money
- Control = Not showing ad = Lost revenue
- The 15% control is just enough to measure causal lift

---

## Exploratory Data Analysis (Notebook 1)

### Key Findings:

1. **Outcome Distribution**:
   - Conversion: 0.29% (extremely rare)
   - Visit: 4.70%
   - Exposure: 3.06%

2. **Average Treatment Effect (ATE)**:
   - Control conversion rate: 0.194%
   - Treatment conversion rate: 0.309%
   - Absolute ATE: +0.115 percentage points
   - Relative Lift: ~59%

3. **Randomization Check**: Feature distributions overlap between treatment and control groups, confirming proper randomization.

4. **Heterogeneous Effects**: Uplift varies across feature segments (e.g., by f0 quartiles), suggesting potential for targeted interventions.

---

## Uplift Modeling (Notebook 2)

### Models Implemented

| Model | Description | Complexity |
|-------|-------------|------------|
| **T-Learner** | Two separate models for treatment & control | Low |
| **S-Learner** | Single model with treatment as feature | Low |
| **X-Learner** | 4-stage approach with counterfactual imputation | High |

### Results

| Model | Qini Coefficient | AUUC | Rank |
|-------|------------------|------|------|
| **T-Learner** 🏆 | 35.19 | 0.00339 | 1st |
| S-Learner | 19.25 | 0.00282 | 2nd |
| X-Learner | 6.62 | 0.00261 | 3rd |

### Why T-Learner Won (Surprising!)

X-Learner is designed for imbalanced treatment/control splits, yet T-Learner performed best. Reasons:

1. **Sufficient Sample Size**: Even with 15% control, we have ~45,000 control samples. This is enough for T-Learner's control model to learn effectively.

2. **X-Learner's Complexity Backfires**: X-Learner has 4 training stages:
   - Stage 1: Train outcome models
   - Stage 2: Impute counterfactuals
   - Stage 3: Train effect models on imputed values
   - Stage 4: Blend with propensity weighting
   
   Each stage introduces errors that compound with noisy features.

3. **Variance-Bias Tradeoff**: With adequate samples, T-Learner's simplicity (lower variance) beats X-Learner's sophistication.

4. **Note on Stage 4 Propensity Weighting**: In an RCT, this is NOT for correcting selection bias (none exists). It's a variance-reduction mechanism that blends CATE estimates based on data availability, not treatment likelihood.

**Rule of Thumb**: X-Learner excels when control < 5% or < 10k samples. With sufficient data, simpler models often win.

---

## Customer Segmentation (Notebook 5)

### Segmentation Rules (Order Matters!)

| Segment | Rule | Rationale |
|---------|------|-----------|
| **Sleeping Dogs** | uplift < -0.5% | Treatment hurts - check first for safety |
| **Sure Things** | baseline > 50% | Will convert anyway - save budget |
| **Persuadables** | uplift > 0.1% | Positive incremental value from treatment |
| **Lost Causes** | Everything else | Low or no uplift |

### Distribution

| Segment | Count | Percentage | Mean Uplift |
|---------|-------|------------|-------------|
| Persuadables | 22,515 | 25.0% | Positive |
| Sure Things | 19,977 | 22.2% | Variable |
| Lost Causes | 24,640 | 27.4% | Near zero |
| Sleeping Dogs | 22,868 | 25.4% | Negative |

---

## Evaluation Metrics

### Qini Coefficient
- Measures the area between the model's Qini curve and random targeting
- Higher = better at ranking Persuadables first
- Interpretation: "How many extra conversions do we get by using the model vs. random targeting?"

### AUUC (Area Under Uplift Curve)
- Average uplift across all targeting thresholds
- Higher = better overall uplift prediction

### Top K% Uplift
- Most actionable business metric
- Shows actual uplift when targeting top K% of users by predicted uplift
- Helps answer: "How many users should we target to maximize ROI?"

---

## Four Types of Users (Industry Standard)

| Type | Without Ad | With Ad | Action |
|------|------------|---------|--------|
| **Persuadables** | Won't buy | Will buy | Target these! |
| **Sure Things** | Will buy | Will buy | Don't waste ad $ |
| **Lost Causes** | Won't buy | Won't buy | Don't waste ad $ |
| **Sleeping Dogs** | Will buy | Won't buy | Avoid! |

Uplift modeling identifies **Persuadables** — users whose behavior changes because of the treatment.

---

## Business Recommendations

1. **Deploy T-Learner** for production uplift scoring
2. **Target top 10-20%** of users by predicted uplift (highest ROI)
3. **Monitor with holdout tests** — validate model predictions with real A/B tests
4. **Retrain periodically** — user behavior changes over time

---

## Project Structure

```
criteo-uplift/
├── streamlit_app/
│   └── app_v2.py                    # Interactive dashboard
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_uplift_modeling.ipynb     # Model training & evaluation
│   ├── 03_causal_forest.ipynb       # Causal Forest experiments
│   ├── 04_advanced_evaluation.ipynb # Qini curves, AUUC
│   └── 05_customer_profiles.ipynb   # Segmentation & SHAP
├── visualizations/
│   ├── data/                        # Pre-computed CSVs
│   └── images/                      # Saved plots
├── data/
│   └── raw/criteo_uplift.parquet    # Full dataset (gitignored)
├── models/                          # Saved model artifacts
├── MODEL_CARD.md                    # Model documentation
├── DATA_DICTIONARY.md               # Feature documentation
├── FINDINGS.md                      # This file
├── LICENSE                          # MIT License
└── README.md
```

---

## Key Learnings

1. **Simpler models can win**: Despite X-Learner being designed for imbalanced data, T-Learner won with sufficient samples.

2. **Sample size matters more than method**: ~45k control samples were enough to make T-Learner effective.

3. **Complexity has costs**: Each additional modeling stage introduces potential errors that compound.

4. **Domain understanding is crucial**: Understanding that this is an advertising RCT explains the unusual 85/15 split.

5. **Metrics matter**: Traditional ML metrics (accuracy, AUC) don't apply. Use Qini, AUUC, and business-focused Top K% analysis.

6. **Uplift vs Propensity**: Propensity models predict who will convert. Uplift models predict who will convert *because of* the treatment. That's the key difference.

---

## References

1. **Diemert et al. (2018)** - "A Large Scale Benchmark for Uplift Modeling" - [Paper](https://bitlater.github.io/files/large-scale-benchmark_comAH.pdf)
2. **Künzel et al. (2019)** - "Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning" (X-Learner paper)
3. **CausalML Documentation** - [github.com/uber/causalml](https://github.com/uber/causalml)
4. **Criteo Dataset** - [huggingface.co/datasets/criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift)
