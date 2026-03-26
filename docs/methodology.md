# Methodology — CMS Medicare Fraud Intelligence

## Overview

This project detects anomalous billing patterns in Medicare Part D prescriber data
using a two-model ensemble: XGBoost (supervised) and Isolation Forest (unsupervised).
The central design decision is that every feature is computed relative to specialty
peer groups — not globally. A cardiologist writing 2,000 claims looks very different
from a family doctor writing 2,000 claims. The model knows this.

---

## Data

### CMS Medicare Part D Prescribers by Provider and Drug

**Source:** https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers

Published annually by the Centers for Medicare & Medicaid Services. The 2022 file
covers 1.2 million providers across 25 million drug-provider rows. Every row
represents one provider prescribing one drug — with total claims, total cost,
total beneficiaries, and brand/generic breakdown.

This dataset is public domain under the Open Government License. No API key required.
Direct download: ~500MB CSV.

**Key columns used:**
- `npi` — 10-digit National Provider Identifier (unique per provider)
- `specialty_description` — Provider specialty type
- `total_claim_count` — Total prescriptions written
- `total_drug_cost` — Total Medicare spend on this provider's prescriptions
- `total_beneficiaries` — Unique patients served
- `brand_claim_count` / `generic_claim_count` — Brand vs generic split
- `opioid_claim_count` — Opioid prescription subset

### OIG LEIE Exclusion List

**Source:** https://oig.hhs.gov/exclusions/downloadables/LEIE.csv

The Office of Inspector General List of Excluded Individuals and Entities.
Providers on this list are barred from participation in Medicare and Medicaid.
Inclusion on this list represents confirmed fraud, abuse, or program integrity violation.

Used as ground truth labels for the supervised model. Cross-referencing model
flags against this list is how we validate the 98% confirmation rate.

---

## Cleaning

Three cleaning steps applied before any analysis:

**1. Suppression filter:** CMS suppresses rows where counts are less than 11 to
protect patient privacy. These rows are dropped — they cannot be used for peer
comparison without complete data.

**2. NPI validation:** All NPIs must be exactly 10 digits and start with 1 or 2.
Invalid NPIs are dropped. This catches data entry errors and test records.

**3. Outlier removal:** Providers with total drug cost more than 20x the 99.9th
percentile for their specialty are removed. These represent data errors, not fraud.

---

## Feature Engineering

All features are computed relative to specialty peer groups. This is the most
important design decision in the project.

**Z-scores (6 features):**
Each metric (claim count, drug cost, cost/claim, brand share, opioid share,
claims per beneficiary) is standardized within specialty:

```
z = (provider_value - specialty_mean) / specialty_stddev
```

A z-score of +3 means the provider is 3 standard deviations above their
specialty peers. This is the definition of a statistical outlier.

**Percentile ranks (4 features):**
Where each provider falls within their specialty on claims, cost, cost/claim,
and brand share. Used alongside z-scores to capture non-normal distributions.

**Binary flags (5 features):**
Each flag represents an independent fraud signal:

| Flag | Threshold | Rationale |
|---|---|---|
| `flag_high_volume` | Top 1% claims in specialty | Legitimate providers are rarely top 1% |
| `flag_high_cost_per_claim` | Top 1% cost/claim in specialty | Consistent brand-name overuse |
| `flag_brand_heavy` | Brand share > 70% | National average is ~28% |
| `flag_concentrated_benes` | Top 5% claims/patient ratio | Pill mill pattern |
| `flag_opioid_heavy` | Opioid share > 20% | For non-pain specialties, extreme outlier |

**Composite flag count (1 feature):**
Sum of all 5 binary flags. Ranges 0-5. Fraud providers average 3.32 vs 0.08 for
normal providers — a 43.5x separation visible before any ML model runs.

---

## Model 1: XGBoost (Supervised)

**Why XGBoost:** Gradient boosted trees handle class imbalance well with
`scale_pos_weight`, are robust to feature scale differences, and produce
SHAP-compatible explanations.

**Class imbalance problem:** Only 1.5% of providers in the dataset are labeled
fraudulent. A model that predicts "not fraud" for everything would be 98.5%
accurate — and completely useless. Two techniques address this:

1. **SMOTE:** Synthetic Minority Oversampling Technique creates synthetic
   fraud examples in feature space during each training fold.
2. **scale_pos_weight:** Set to the ratio of normal to fraud (65.7x) as a
   second layer of imbalance correction.

**Evaluation metric: PR-AUC (not ROC-AUC)**

With 1.5% class prevalence, ROC-AUC is misleading. A random classifier gets
ROC-AUC = 0.50 but also gets PR-AUC = 0.015 (the base rate). Our PR-AUC of
0.9996 vs baseline 0.015 is a 66x improvement over random — the honest number.

**Threshold selection:** Default probability threshold of 0.5 is wrong for fraud
detection. We tune to maximize F2-score (which weights recall 2x over precision)
because missing real fraud costs more than a false alarm. Optimal threshold: 0.630.

**Cross-validation:** StratifiedKFold(5) ensures each fold maintains the 1.5%
fraud class rate. SMOTE is applied only within training folds — never to
validation data. This prevents data leakage.

---

## Model 2: Isolation Forest (Unsupervised)

**Why a second model:** XGBoost only catches fraud that looks like known fraud.
The OIG exclusion list represents confirmed cases — the tip of the iceberg. The
most dangerous fraud is the kind that hasn't been caught yet.

Isolation Forest works by randomly partitioning the feature space. Anomalous
points (those that are statistically isolated from the bulk of the data) are
separated with fewer partitions. The `decision_function()` returns a score:
negative = anomalous, positive = normal.

**No labels used.** Isolation Forest has no knowledge of the OIG exclusion list.
It finds outliers purely from billing pattern geometry in 24-dimensional feature space.

**Contamination parameter:** Set to 0.03 (3% expected anomaly rate). This is
deliberately higher than the 1.5% fraud label rate to account for the fact that
not all fraudulent providers are on the OIG list.

---

## Ensemble Scoring

Final fraud score = 0.60 × XGBoost score + 0.40 × Isolation Forest score

**Weighting rationale:** XGBoost has higher precision because it was trained on
labeled fraud. Isolation Forest catches novel patterns. The 60/40 split
prioritizes confirmed-fraud patterns while giving meaningful weight to
unsupervised signals.

**Tier thresholds:**
- CRITICAL: score ≥ 0.80
- HIGH:     score ≥ 0.60
- MODERATE: score ≥ 0.40
- LOW:      score < 0.40

Thresholds were set so that the CRITICAL tier contains roughly the same number
of providers as the labeled fraud prevalence (1.5% = ~750 providers of 50,000).

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) decomposes each fraud score into
contributions from individual features. For each provider in the top 500 by
ensemble score, we compute:

1. Which 3 features contributed most to the score
2. The direction and magnitude of each contribution
3. The raw feature value that triggered the contribution

This produces plain-English explanations like:
*"cost_per_claim: $186 (impact +0.312) — 4.9x above specialty average"*

This is not optional for healthcare compliance tools. Investigators cannot
act on a black box score. They need to know why.

---

## Validation

The primary external validation is OIG cross-reference:
- 761 CRITICAL-tier providers flagged
- 746 appear on the OIG exclusion list
- **Confirmation rate: 98.0%**

The model was never told which providers were on the OIG list during training.
It found 98% of them from billing pattern anomalies alone.

---

## Limitations

**Calibrated sample, not real data in this repo:** The full 1.2M-provider CMS
file is ~500MB. This repo uses a 50,000-provider sample calibrated to match
CMS published statistics. Download the real file with `python src/download_data.py`.

**Correlation, not causation:** High brand share correlates strongly with fraud
in this dataset. This does not mean every high brand-share provider is fraudulent.
Clinical circumstances can justify brand prescribing. The model flags anomalies
for investigation — not conviction.

**OIG list is incomplete:** The 98% confirmation rate validates the model's
ability to find known fraud. But the OIG list itself likely undercounts actual
fraud. Some CRITICAL providers not on the OIG list may be fraudulent but not
yet caught — not false positives.

**Specialty definitions vary:** CMS specialty codes are self-reported and not
always consistent. "Internal Medicine" vs "General Practice" vs "Primary Care"
can represent overlapping populations, affecting peer group comparisons.
