# Limitations & Caveats

## On the Data

### This repo uses a calibrated sample, not the real CMS file

The full CMS Medicare Part D 2022 file is ~500MB with 1.2 million providers.
This repo uses a 50,000-provider sample calibrated to match CMS published statistics:
specialty distribution, state distribution, cost benchmarks, and fraud prevalence.

To run on the real data:
```bash
python src/download_data.py   # downloads ~500MB from data.cms.gov
python run_pipeline.py
```

Real download URL:
```
https://data.cms.gov/sites/default/files/2024-04/MUP_DPR_RY24_P04_V10_DY22_NPIBN.csv
```

### The OIG exclusion list is incomplete by definition

The OIG LEIE list contains providers who were *caught* and *excluded*.
It does not contain providers who are committing fraud but haven't been caught,
providers under active investigation, or providers who settled without exclusion.

The 98% confirmation rate means: of providers flagged CRITICAL, 98% were already
on the OIG list. It does not mean: every provider flagged CRITICAL is definitely
fraudulent. Some may be legitimate outliers with unusual but explainable billing.

### Brand drug share has clinical explanations

A 70%+ brand drug share triggers a flag. But some legitimate situations produce
high brand share:

- Oncology prescribers treating patients on brand-name targeted therapies
- Providers in markets where generic availability is limited
- Providers serving patient populations with documented generic intolerances
- Recent prescribers still building formulary familiarity

The model treats brand share as a probabilistic signal, not a verdict.

---

## On the Model

### High PR-AUC is partly a function of calibrated data

The 0.9996 PR-AUC on calibrated data means the synthetic fraud cases are well-separated
from synthetic normal cases — because they were generated to be separated. On the real
1.2M CMS file, expect lower PR-AUC (likely 0.70-0.85 range) because real fraud
is more heterogeneous and harder to distinguish from legitimate outliers.

The feature engineering (43.5x flag separation) is the part that would transfer
most cleanly to real data. The model score itself is calibrated to synthetic patterns.

### Isolation Forest is not validated against ground truth

Isolation Forest is fully unsupervised. It identifies statistical outliers.
Whether those outliers represent fraud, unusual legitimate practice, or data anomalies
requires human review. The 15 providers flagged only by Isolation Forest (not XGBoost)
are the highest-uncertainty cases in the dataset.

### SHAP explanations are post-hoc

SHAP values explain what the model *did*, not necessarily why fraud occurs.
If the model learned a spurious correlation in the training data, SHAP will
faithfully explain that spurious correlation. The explanations are useful for
directing investigation — they are not evidence of fraud.

---

## On the Use Case

### This tool is for investigation prioritization, not prosecution

Output from this model should be treated as:
- A prioritization tool for directing limited investigative resources
- A screening tool to surface cases warranting further manual review
- A pattern analysis tool to understand fraud typologies in claims data

It should not be treated as:
- Proof of fraud
- Sufficient basis for provider exclusion
- A replacement for clinical or legal review

### Healthcare data requires special handling

CMS data, even aggregated and publicly released, touches on patient care patterns.
Any production deployment of a system like this should comply with applicable
healthcare data governance requirements and involve appropriate legal review.

---

## On Reproducibility

All results in this repo are reproducible by running `python run_pipeline.py`.

The random seed is fixed (`numpy.random.seed(42)`, XGBoost `random_state=42`,
SMOTE `random_state=42`) so every run produces identical results.

The calibrated sample generation script is deterministic given the same seed.
Specialty benchmarks, z-scores, flags, model scores, and SHAP explanations
will all match across runs on the same machine and Python version.
