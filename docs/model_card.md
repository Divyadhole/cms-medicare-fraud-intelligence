# Model Card — CMS Medicare Fraud Intelligence

## Model Details
- **Type:** Ensemble (XGBoost + Isolation Forest)
- **Version:** 1.0
- **Date:** March 2024
- **Author:** Divya Dhole, MS Data Science, University of Arizona

## Intended Use
- **Primary use:** Prioritization of Medicare Part D prescriber billing patterns for investigative review
- **Users:** Healthcare fraud investigators, CMS program integrity analysts, compliance teams
- **Out-of-scope:** Direct exclusion decisions, legal proceedings, clinical judgments

## Training Data
- CMS Medicare Part D Prescribers by Provider and Drug, 2022
- 50,000-provider calibrated sample (full file: 1.2M providers)
- Fraud labels: OIG LEIE Exclusion List cross-reference
- Class balance: 98.5% normal / 1.5% fraud

## Performance
| Metric | Value |
|---|---|
| PR-AUC | 0.9996 |
| ROC-AUC | 1.0000 |
| Optimal threshold | 0.630 |
| Precision (fraud) | 0.98 |
| Recall (fraud) | 1.00 |
| OIG confirmation (CRITICAL tier) | 98.0% |

## Caveats
- Calibrated sample performance may not directly transfer to full CMS dataset
- OIG list is incomplete — confirmed fraud undercounts actual fraud
- High brand share has legitimate clinical explanations
- See docs/limitations.md for full discussion

## Fairness Considerations
This model is trained on billing patterns, not provider demographics.
Specialty is used only to define peer groups, not as a protected characteristic.
No race, gender, age, or other demographic features are used.
