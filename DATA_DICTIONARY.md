# Data Dictionary

## Criteo Uplift Dataset

This document describes the features and labels in the Criteo Uplift Dataset.

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `f0` | Float | Anonymized user/context feature |
| `f1` | Float | Anonymized user/context feature |
| `f2` | Float | Anonymized user/context feature |
| `f3` | Float | Anonymized user/context feature |
| `f4` | Float | Anonymized user/context feature |
| `f5` | Float | Anonymized user/context feature |
| `f6` | Float | Anonymized user/context feature |
| `f7` | Float | Anonymized user/context feature |
| `f8` | Float | Anonymized user/context feature |
| `f9` | Float | Anonymized user/context feature |
| `f10` | Float | Anonymized user/context feature |
| `f11` | Float | Anonymized user/context feature |

**Note:** All features have been randomly projected for privacy. Original feature meanings are not disclosed.

## Treatment & Labels

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `treatment` | Binary | 0, 1 | Whether user was in treatment group (shown ad) |
| `conversion` | Binary | 0, 1 | Whether user converted (purchased) |
| `visit` | Binary | 0, 1 | Whether user visited the website |
| `exposure` | Binary | 0, 1 | Whether user was exposed to the ad |

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Samples | ~25 million |
| Treatment Ratio | 85% treatment / 15% control |
| Conversion Rate | ~0.05% |
| Visit Rate | ~4.9% |
| ATE (Conversion) | +0.11 percentage points |

## Data Collection

- **Source:** Criteo incrementality tests
- **Method:** Randomized Control Trial (RCT)
- **Period:** Multiple campaigns aggregated
- **Privacy:** Sub-sampled non-uniformly, features randomly projected

## Usage Notes

1. **Treatment Assignment:** Random (T ⊥ X), enabling causal interpretation
2. **Imbalanced Classes:** Very low conversion rate (~0.05%)
3. **Large Scale:** Enables statistical significance in uplift evaluation
4. **Anonymized:** Cannot recover original user context

## Citation

```
Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R. (2018). 
A Large Scale Benchmark for Uplift Modeling. 
In Proceedings of AdKDD & TargetAd (ADKDD'18).
https://bitlater.github.io/files/large-scale-benchmark_comAH.pdf
```
