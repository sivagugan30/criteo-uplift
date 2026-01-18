# Model Card: T-Learner Uplift Model

## Model Details

| Field | Value |
|-------|-------|
| **Model Type** | T-Learner (Two-Model Approach) |
| **Base Classifier** | XGBoost (n_estimators=100, max_depth=5) |
| **Task** | Uplift Modeling / Conditional Average Treatment Effect (CATE) Estimation |
| **Framework** | CausalML + XGBoost |
| **Training Date** | January 2025 |
| **Author** | Sivagugan Jayachandran |

## Intended Use

- **Primary Use:** Predict incremental conversion lift from advertising exposure
- **Users:** Marketing teams, data scientists working on campaign optimization
- **Out of Scope:** Real-time bidding, individual user targeting without additional validation

## Training Data

| Field | Value |
|-------|-------|
| **Dataset** | Criteo Uplift Dataset |
| **Source** | [HuggingFace](https://huggingface.co/datasets/criteo/criteo-uplift) |
| **Size** | ~25 million samples |
| **Type** | Randomized Control Trial (RCT) |
| **Treatment Ratio** | 85% treatment / 15% control |
| **Features** | 12 anonymized features (f0-f11) |

## Evaluation Metrics

| Metric | T-Learner | S-Learner | X-Learner |
|--------|-----------|-----------|-----------|
| **Qini Coefficient** | **35.19** | 19.25 | 6.62 |
| **AUUC** | **0.339%** | 0.282% | 0.261% |
| **Mean Predicted Uplift** | 0.177% | 0.102% | 0.121% |

**Winner: T-Learner** - Highest Qini Coefficient and AUUC

## How It Works

T-Learner trains two separate models:
1. **Control Model:** P(conversion | no treatment)
2. **Treatment Model:** P(conversion | treatment)

**Predicted Uplift = Treatment Model - Control Model**

## Limitations

- **Anonymized Features:** Cannot interpret what drives uplift
- **Dataset Specific:** Trained on Criteo data, may not generalize to other domains
- **Static Model:** Does not account for temporal changes in user behavior
- **No Hyperparameter Tuning:** Uses default XGBoost parameters

## Ethical Considerations

- Model should not be used for discriminatory targeting
- Uplift predictions should be validated with A/B tests before production deployment
- Consider user privacy when implementing targeting strategies

## Citation

If using this model or analysis, please cite the original dataset:

```
Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R. (2018). 
A Large Scale Benchmark for Uplift Modeling. 
In Proceedings of AdKDD & TargetAd (ADKDD'18).
```
