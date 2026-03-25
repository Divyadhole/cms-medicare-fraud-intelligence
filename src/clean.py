"""
src/clean.py
Cleans and validates CMS Part D data.
Handles suppressed rows, invalid NPIs, outlier costs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path("data/raw/cms_partd_2022_sample.csv")
OUT  = Path("data/processed/cms_cleaned.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "npi","nppes_provider_state","specialty_description",
    "total_claim_count","total_drug_cost","total_beneficiaries",
    "brand_claim_count","generic_claim_count","opioid_claim_count",
]


def load_and_clean(path: Path = RAW) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path, dtype={"npi": str})
    print(f"  Raw rows: {len(df):,}")

    # 1. Drop rows with suppressed CMS data (CMS suppresses counts < 11)
    for col in ["total_claim_count","total_beneficiaries"]:
        df = df[pd.to_numeric(df[col], errors="coerce").notna()]
        df = df[df[col].astype(float) >= 11]
    print(f"  After suppression filter: {len(df):,}")

    # 2. Validate NPI format (10 digits, starts with 1 or 2)
    df["npi"] = df["npi"].astype(str).str.strip()
    df = df[df["npi"].str.match(r"^[12]\d{9}$")]
    print(f"  After NPI validation: {len(df):,}")

    # 3. Cast numerics
    num_cols = ["total_claim_count","total_drug_cost","total_beneficiaries",
                "brand_claim_count","generic_claim_count","opioid_claim_count",
                "total_30_day_fill_count"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. Remove extreme cost outliers (> 99.9th percentile per specialty)
    p999 = df.groupby("specialty_description")["total_drug_cost"].transform(
        lambda x: x.quantile(0.999))
    extreme = df["total_drug_cost"] > p999 * 20   # 20x the 99.9th — clearly bad data
    n_extreme = extreme.sum()
    if n_extreme > 0:
        print(f"  Removing {n_extreme} extreme cost outliers")
        df = df[~extreme]

    # 5. Standardize specialty names
    df["specialty_description"] = df["specialty_description"].str.strip().str.title()

    # 6. Add derived columns
    df["cost_per_claim"] = (df["total_drug_cost"] /
                             df["total_claim_count"].replace(0, np.nan)).round(2)
    df["cost_per_beneficiary"] = (df["total_drug_cost"] /
                                   df["total_beneficiaries"].replace(0, np.nan)).round(2)
    df["brand_share"] = (df["brand_claim_count"] /
                          df["total_claim_count"].replace(0, np.nan)).round(4)
    df["opioid_share"] = (df["opioid_claim_count"] /
                           df["total_claim_count"].replace(0, np.nan)).round(4)
    df["claims_per_bene"] = (df["total_claim_count"] /
                               df["total_beneficiaries"].replace(0, np.nan)).round(2)

    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def save_clean(df: pd.DataFrame) -> None:
    df.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== CLEAN DATA SUMMARY ===")
    print(f"Total providers:     {len(df):,}")
    print(f"Total drug cost:     ${df['total_drug_cost'].sum()/1e9:.2f}B")
    print(f"Avg claims/provider: {df['total_claim_count'].mean():.0f}")
    print(f"Avg cost/claim:      ${df['cost_per_claim'].mean():.0f}")
    print(f"Avg brand share:     {df['brand_share'].mean()*100:.1f}%")
    print(f"\nTop 5 specialties:")
    print(df["specialty_description"].value_counts().head())
    if "is_fraud_label" in df.columns:
        fraud = df["is_fraud_label"].sum()
        print(f"\nFraud labels:        {fraud:,} ({fraud/len(df)*100:.2f}%)")


if __name__ == "__main__":
    df = load_and_clean()
    print_summary(df)
    save_clean(df)
