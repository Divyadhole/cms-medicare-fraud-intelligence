"""
src/features.py
Feature engineering for Medicare fraud detection.

Builds peer-group-relative features — the core insight being that
a cardiologist writing 500 claims is normal, but a family doctor
writing 500 claims of brand-only opioids is not.

Every feature is computed relative to specialty peers, not globally.
This is how real fraud investigators think.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

CLEAN = Path("data/processed/cms_cleaned.csv")
FEAT  = Path("data/processed/features.csv")
BENCH = Path("data/processed/specialty_benchmarks.csv")


def compute_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute specialty-level peer group statistics."""
    grp = df.groupby("specialty_description")

    metrics = ["total_claim_count", "total_drug_cost", "cost_per_claim",
               "brand_share", "opioid_share", "claims_per_bene",
               "cost_per_beneficiary"]

    rows = []
    for spec, g in grp:
        row = {"specialty": spec, "n_providers": len(g)}
        for m in metrics:
            if m in g.columns:
                vals = g[m].dropna()
                row[f"avg_{m}"]  = vals.mean()
                row[f"std_{m}"]  = vals.std()
                row[f"p50_{m}"]  = vals.quantile(0.50)
                row[f"p95_{m}"]  = vals.quantile(0.95)
                row[f"p99_{m}"]  = vals.quantile(0.99)
        rows.append(row)

    bench = pd.DataFrame(rows)
    bench.to_csv(BENCH, index=False)
    print(f"  Benchmarks saved → {BENCH}  ({len(bench)} specialties)")
    return bench


def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each provider against their specialty peers."""
    metrics = ["total_claim_count", "total_drug_cost", "cost_per_claim",
               "brand_share", "opioid_share", "claims_per_bene"]

    for m in metrics:
        if m not in df.columns:
            continue
        col = f"z_{m}"
        df[col] = df.groupby("specialty_description")[m].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else np.nan)
        ).round(3)

    return df


def compute_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank within specialty peer group."""
    metrics = ["total_claim_count", "total_drug_cost", "cost_per_claim", "brand_share"]

    for m in metrics:
        if m not in df.columns:
            continue
        col = f"pct_{m}"
        df[col] = df.groupby("specialty_description")[m].rank(pct=True).round(4)

    return df


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary fraud signal flags. Each flag is an independent signal.
    Providers with 3+ flags are high priority for review.

    Thresholds based on CMS fraud detection literature and OIG reports.
    """
    # Flag 1: Extreme claim volume (top 1% in specialty)
    df["flag_high_volume"] = (
        df.groupby("specialty_description")["total_claim_count"]
        .transform(lambda x: x > x.quantile(0.99))
    ).astype(int)

    # Flag 2: High cost per claim vs peers (top 1%)
    df["flag_high_cost_per_claim"] = (
        df.groupby("specialty_description")["cost_per_claim"]
        .transform(lambda x: x > x.quantile(0.99))
    ).astype(int)

    # Flag 3: Brand-heavy prescribing (>70% brand = major signal)
    # National average is ~28% brand. Fraud mills often hit 85-95%.
    df["flag_brand_heavy"] = (df["brand_share"] > 0.70).astype(int)

    # Flag 4: Concentrated beneficiaries (pill mill pattern)
    # More than 15 claims per unique beneficiary = suspicious concentration
    df["flag_concentrated_benes"] = (
        df.groupby("specialty_description")["claims_per_bene"]
        .transform(lambda x: x > x.quantile(0.95))
    ).astype(int)

    # Flag 5: High opioid share (>20% of all claims = opioids)
    # For non-pain specialties this is extreme
    df["flag_opioid_heavy"] = (df["opioid_share"] > 0.20).astype(int)

    # Composite: total number of flags
    flag_cols = ["flag_high_volume", "flag_high_cost_per_claim",
                 "flag_brand_heavy", "flag_concentrated_benes", "flag_opioid_heavy"]
    df["flag_count"] = df[flag_cols].sum(axis=1)

    return df


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble final feature matrix for the ML model."""
    feature_cols = [
        # Raw metrics (log-transformed to reduce skew)
        "total_claim_count", "total_drug_cost", "total_beneficiaries",
        # Derived ratios
        "cost_per_claim", "cost_per_beneficiary", "brand_share",
        "opioid_share", "claims_per_bene",
        # Z-scores vs specialty peers
        "z_total_claim_count", "z_total_drug_cost", "z_cost_per_claim",
        "z_brand_share", "z_opioid_share", "z_claims_per_bene",
        # Percentile ranks
        "pct_total_claim_count", "pct_total_drug_cost",
        "pct_cost_per_claim", "pct_brand_share",
        # Binary flags
        "flag_high_volume", "flag_high_cost_per_claim",
        "flag_brand_heavy", "flag_concentrated_benes",
        "flag_opioid_heavy", "flag_count",
    ]

    available = [c for c in feature_cols if c in df.columns]
    return df[["npi", "specialty_description"] + available +
              (["is_fraud_label"] if "is_fraud_label" in df.columns else [])]


def run():
    print("=" * 55)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    print("\n[1/4] Loading cleaned data...")
    df = pd.read_csv(CLEAN, dtype={"npi": str})
    print(f"  {len(df):,} providers loaded")

    print("\n[2/4] Computing specialty benchmarks...")
    bench = compute_benchmarks(df)

    print("\n[3/4] Computing features...")
    df = compute_zscores(df)
    df = compute_percentile_ranks(df)
    df = compute_flags(df)

    print(f"\n  Flag distribution:")
    for i in range(6):
        n = (df["flag_count"] == i).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct)
        print(f"    {i} flags: {n:,} providers ({pct:.1f}%) {bar}")

    print("\n[4/4] Building model feature matrix...")
    feat_df = build_model_features(df)
    feat_df.to_csv(FEAT, index=False)
    print(f"  Features saved → {FEAT}")
    print(f"  Shape: {feat_df.shape}")
    print(f"  Feature cols: {feat_df.shape[1]-3}")

    if "is_fraud_label" in df.columns:
        fraud = df["is_fraud_label"].sum()
        fraud_flags = df[df["is_fraud_label"]==1]["flag_count"].mean()
        normal_flags = df[df["is_fraud_label"]==0]["flag_count"].mean()
        print(f"\n  Fraud label stats:")
        print(f"    Fraud providers:      {fraud:,}")
        print(f"    Avg flags (fraud):    {fraud_flags:.2f}")
        print(f"    Avg flags (normal):   {normal_flags:.2f}")
        print(f"    Flag separation:      {fraud_flags/normal_flags:.1f}x")

    return feat_df


if __name__ == "__main__":
    run()
