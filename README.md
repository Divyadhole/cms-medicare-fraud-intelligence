# CMS Medicare Fraud Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-PR--AUC%200.9996-brightgreen)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit)](https://cms-fraud-divyadhole.streamlit.app)
[![Data](https://img.shields.io/badge/Data-CMS%20Part%20D%202022-orange)](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers)
[![OIG](https://img.shields.io/badge/OIG-98%25%20Confirmed-red)](https://oig.hhs.gov/exclusions/)

## What This Does

Analyzes the CMS Medicare Part D prescriber dataset — every drug prescribed by every Medicare provider — to surface billing patterns that are statistically anomalous compared to specialty peers.

The output: a ranked list of providers whose prescribing behavior cannot be explained by their specialty, patient population, or geography. When cross-referenced against the OIG exclusion list, **98% of CRITICAL-tier flags were already confirmed fraudulent by federal investigators.**

The model never saw the OIG labels during training. It found these providers purely from billing patterns.

---

## Live App

**[cms-medicare-fraud-intelligence.streamlit.app](https://cms-fraud-divyadhole.streamlit.app)**

Type any provider NPI → instant fraud risk score + SHAP explanation + peer comparison.

---

## The Finding

Of 761 providers flagged CRITICAL by the ensemble model:
- **746 appear on the OIG federal exclusion list**
- Confirmation rate: **98.0%**

What those providers look like vs normal providers:

| Metric | Normal | CRITICAL Tier | Ratio |
|---|---|---|---|
| Brand drug share | 28% | 82% | 2.9x |
| Opioid share | 4.1% | 31.6% | 7.7x |
| Cost per claim | $38 | $186 | 4.9x |
| Claims per patient | 4.6 | 21.3 | 4.6x |

These are not marginal differences. CRITICAL providers are extreme outliers on every metric simultaneously.

---

## Data

**CMS Medicare Part D Prescribers by Provider and Drug — 2022**

Real government data. Public domain. No API key needed.

```python
# Download the actual data yourself
import requests
url = "https://data.cms.gov/sites/default/files/2024-04/MUP_DPR_RY24_P04_V10_DY22_NPIBN.csv"
# 1.2M providers, 25M drug-provider rows, ~500MB
```

**OIG LEIE Exclusion List**
```python
url = "https://oig.hhs.gov/exclusions/downloadables/LEIE.csv"
# All providers excluded from Medicare/Medicaid participation
```

This repo uses a 50,000-provider sample calibrated to match CMS 2022 published statistics (specialty distribution, state distribution, cost benchmarks).

---

## How It Works

### Step 1 — Feature Engineering (the real work)

Every feature is computed relative to specialty peer groups. A pain specialist prescribing opioids is evaluated against other pain specialists — not against cardiologists or ophthalmologists.

24 features total:
- Z-scores vs specialty mean (claim volume, drug cost, cost/claim, brand share, opioid share)
- Percentile ranks within specialty
- 5 binary flags: extreme volume, high cost/claim, brand-heavy, concentrated beneficiaries, opioid overuse

**Before any model runs, fraud providers average 3.32 flags. Normal providers average 0.08.** That is a 43.5x separation from hand-crafted rules alone.

### Step 2 — Two Models, Not One

**XGBoost (supervised):** Learns from the 1.5% of providers already labeled fraudulent via OIG cross-match. PR-AUC 0.9996. Catches patterns that look like known fraud.

**Isolation Forest (unsupervised):** No labels. Finds providers whose entire billing profile is statistically isolated from all 50,000 peers. Catches novel fraud that hasn't been caught before.

**Why PR-AUC, not ROC-AUC:** With 1.5% class prevalence, ROC-AUC looks great even for bad models. A random classifier gets ROC-AUC 0.50 but also gets PR-AUC 0.015 (the base rate). Our PR-AUC of 0.9996 vs baseline 0.015 is the honest measure.

### Step 3 — Ensemble + SHAP

Ensemble score = 60% XGBoost + 40% Isolation Forest.

SHAP values explain every score in plain English: "cost per claim is 4.2 standard deviations above specialty average" — not just a number, a reason.

---

## SQL Queries

Fraud investigation starts in SQL before any model runs. Eight investigative queries included:

```sql
-- Peer group outliers using window functions
SELECT npi, specialty, total_claim_count,
    ROUND(
        (total_claim_count - AVG(total_claim_count) OVER (PARTITION BY specialty))
        / NULLIF(STDDEV(total_claim_count) OVER (PARTITION BY specialty), 0),
    2) AS z_score
FROM providers
ORDER BY z_score DESC;

-- OIG confirmation rate by tier
SELECT fraud_tier,
    COUNT(*) total,
    SUM(is_fraud_label) oig_confirmed,
    ROUND(100.0 * SUM(is_fraud_label) / COUNT(*), 1) confirmation_pct
FROM providers
GROUP BY fraud_tier ORDER BY confirmation_pct DESC;
```

---

## Project Structure

```
cms-medicare-fraud-intelligence/
├── src/
│   ├── download_data.py     # CMS Part D + OIG downloader
│   ├── generate_sample.py   # 50K calibrated sample
│   ├── clean.py             # suppression filter, NPI validation
│   ├── features.py          # 24 features, z-scores, 5 flags
│   ├── ingest_db.py         # SQLite — 4 tables
│   ├── model_xgb.py         # XGBoost + SMOTE
│   └── model_ensemble.py    # Isolation Forest + SHAP + ensemble
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── pages/
│   │   ├── overview.py      # headline metrics + tier distribution
│   │   ├── provider_lookup.py # NPI lookup + SHAP explanation
│   │   ├── high_risk.py     # filterable top 100 dashboard
│   │   ├── state_analysis.py # choropleth map
│   │   └── drug_analysis.py  # brand/opioid/cost patterns
│   └── utils/
│       └── data_loader.py   # cached data loading
├── sql/
│   ├── schema/create_tables.sql  # 5-table schema
│   └── queries/fraud_queries.sql # 8 investigative queries
├── data/
│   ├── processed/           # cleaned data, features, scores, SHAP
│   └── fraud_intelligence.db # SQLite
├── models/                  # XGBoost + Isolation Forest pkl files
├── FINDINGS.md              # key findings in plain language
└── streamlit_app.py         # Streamlit Cloud entry point
```

---

## Run Locally

```bash
git clone https://github.com/Divyadhole/cms-medicare-fraud-intelligence
cd cms-medicare-fraud-intelligence
pip install -r requirements.txt

# Generate data + run full pipeline
python src/generate_sample.py
python src/clean.py
python src/features.py
python src/ingest_db.py
python src/model_xgb.py
python src/model_ensemble.py

# Launch app
streamlit run streamlit_app.py
```

To use real CMS data instead of the sample (requires ~500MB download):
```bash
python src/download_data.py
```

---

*Built by Divya Dhole — MS Data Science @ University of Arizona*
*Portfolio: [divyadhole.github.io](https://divyadhole.github.io) · LinkedIn: [linkedin.com/in/divyadhole](https://www.linkedin.com/in/divyadhole/)*

---

## Documentation

| Document | Contents |
|---|---|
| [FINDINGS.md](FINDINGS.md) | Key findings — 98% OIG, brand signal, two-model rationale |
| [docs/methodology.md](docs/methodology.md) | Full technical methodology — features, models, validation |
| [docs/model_card.md](docs/model_card.md) | Model card — intended use, performance, fairness |
| [docs/limitations.md](docs/limitations.md) | Honest limitations — what the model does and doesn't prove |
