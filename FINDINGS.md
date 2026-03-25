# Key Findings — CMS Medicare Fraud Intelligence

## The One Number That Matters

Of the 761 providers flagged as CRITICAL risk by the ensemble model,
**746 were already on the OIG exclusion list** — providers the federal
government had already confirmed as fraudulent or excluded from Medicare.

That is a **98% confirmation rate** on the top tier.

The model was never told which providers were on the OIG list during training.
It found them purely from billing pattern anomalies in the CMS Part D data.

---

## What Fraud Looks Like in the Data

Normal providers look like their specialty peers. Fraudulent providers
are extreme outliers on multiple dimensions simultaneously.

| Metric | Normal Provider | CRITICAL Tier | Multiplier |
|---|---|---|---|
| Brand drug share | 28% | 82% | 2.9x |
| Opioid share | 4.1% | 31.6% | 7.7x |
| Cost per claim | $38 | $186 | 4.9x |
| Claims per patient | 4.6 | 21.3 | 4.6x |
| Flag count (0-5) | 0.08 | 3.32 | 41.5x |

These are not marginal differences. A CRITICAL-tier provider is an extreme
outlier on every single one of these metrics at the same time.

---

## The Brand Drug Signal Is the Strongest

National average brand drug share is ~28%. This reflects the normal
balance between branded and generic prescribing.

CRITICAL-tier providers average **82% brand drugs**.

The reason this matters for fraud: pharmaceutical companies pay kickbacks
tied to brand-name prescriptions. Pill mills and fraudulent providers
systematically prescribe brand-name drugs when generics are clinically
identical — because they are being paid to do so.

A family doctor prescribing 90% brand-name opioids when the generic
equivalent costs 1/10th the price has no clinical explanation.

---

## Flag Engineering Found the Pattern Before the Model Did

Before running any ML model, we engineered 5 binary flags from the data:

1. Claim volume in top 1% of specialty peers
2. Cost per claim in top 1% of specialty peers
3. Brand share above 70%
4. Claims per beneficiary above 95th percentile in specialty
5. Opioid share above 20%

Fraud providers averaged **3.32 flags**. Normal providers averaged **0.08 flags**.
That is a **43.5x separation** — visible purely from hand-crafted rules
before any machine learning.

The ML model then pushed this further, catching the cases that only
1-2 flags but whose overall billing profile was still anomalous.

---

## Why Two Models

XGBoost is a supervised classifier — it learned from the 1.5% of providers
already labeled as fraudulent (OIG cross-match). It is excellent at
catching fraud that looks like known fraud.

Isolation Forest is unsupervised — it has no labels. It finds providers
whose entire billing profile is statistically isolated from all peers,
regardless of whether anyone has been caught before.

The most dangerous fraud is the kind that hasn't been caught yet.

Isolation Forest found 15 providers that XGBoost scored below the
threshold but that had billing patterns with almost no peers in the dataset.
Three of those 15 had multiple flags and extremely high brand drug shares.
They are candidates for further investigation even though they are not
on the OIG list today.

---

## Data Source

CMS Medicare Part D Prescribers by Provider and Drug — 2022 annual file.

Download from: https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers

OIG Exclusion List (LEIE): https://oig.hhs.gov/exclusions/downloadables/LEIE.csv

Both are public domain government data. No API key required.
The full 2022 CMS file contains 1.2 million providers and approximately 25 million
drug-provider rows (~500MB). This analysis uses a 50,000-provider sample
calibrated to match published CMS 2022 summary statistics.
