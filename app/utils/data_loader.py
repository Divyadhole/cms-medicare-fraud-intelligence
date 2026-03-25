"""
app/utils/data_loader.py
Loads and caches all data for the Streamlit app.
Uses st.cache_data so the DB is only read once per session.
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DB      = Path("data/fraud_intelligence.db")
MODELS  = Path("models")
SCORES  = Path("data/processed/ensemble_scores.csv")
SHAP    = Path("data/processed/shap_reasons.csv")


@staticmethod
def _get_conn():
    return sqlite3.connect(DB)


def load_providers() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql("""
        SELECT p.*,
               f.flag_count, f.flag_high_volume, f.flag_high_cost_per_claim,
               f.flag_brand_heavy, f.flag_concentrated_benes, f.flag_opioid_heavy,
               f.z_total_claim_count, f.z_cost_per_claim,
               f.z_brand_share, f.z_opioid_share
        FROM providers p
        LEFT JOIN provider_features f ON p.npi = f.npi
    """, conn, dtype={"npi": str})
    conn.close()

    # Merge ensemble scores
    if SCORES.exists():
        scores = pd.read_csv(SCORES, dtype={"npi": str})
        df = df.merge(scores[["npi","fraud_score_xgb","fraud_score_iso",
                               "fraud_score_ensemble","fraud_tier"]],
                      on="npi", how="left", suffixes=("_old",""))
        # Drop duplicate columns from old DB scores
        for col in ["fraud_score_xgb_old","fraud_score_iso_old",
                    "fraud_score_ensemble_old","fraud_tier_old"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    return df


def load_shap() -> pd.DataFrame:
    if SHAP.exists():
        return pd.read_csv(SHAP, dtype={"npi": str})
    return pd.DataFrame()


def load_benchmarks() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM specialty_benchmarks", conn)
    conn.close()
    return df


def load_models():
    model     = joblib.load(MODELS / "xgb_fraud.pkl")
    iso       = joblib.load(MODELS / "iso_forest.pkl")
    scaler    = joblib.load(MODELS / "iso_scaler.pkl")
    threshold = joblib.load(MODELS / "xgb_threshold.pkl")
    features  = joblib.load(MODELS / "feature_names.pkl")
    return model, iso, scaler, threshold, features


def get_provider(npi: str, df: pd.DataFrame) -> pd.Series | None:
    match = df[df["npi"] == str(npi).strip()]
    if len(match) == 0:
        return None
    return match.iloc[0]


def tier_color(tier: str) -> str:
    return {
        "CRITICAL": "#ef4444",
        "HIGH":     "#f97316",
        "MODERATE": "#eab308",
        "LOW":      "#22c55e",
    }.get(tier, "#6b7280")


def tier_badge(tier: str) -> str:
    colors = {
        "CRITICAL": ("ef4444", "fff"),
        "HIGH":     ("f97316", "fff"),
        "MODERATE": ("eab308", "000"),
        "LOW":      ("22c55e", "fff"),
    }
    bg, fg = colors.get(tier, ("6b7280", "fff"))
    return f'<span style="background:#{bg};color:#{fg};padding:4px 12px;border-radius:99px;font-weight:700;font-size:13px">{tier}</span>'


FEATURE_DISPLAY = {
    "z_total_claim_count":    "Claim volume vs specialty peers",
    "z_cost_per_claim":       "Cost per claim vs specialty peers",
    "z_brand_share":          "Brand drug usage vs specialty peers",
    "z_opioid_share":         "Opioid prescribing vs specialty peers",
    "flag_high_volume":       "Extreme claim volume (top 1% in specialty)",
    "flag_high_cost_per_claim": "High cost per claim (top 1% in specialty)",
    "flag_brand_heavy":       "Brand-heavy prescribing (>70% brand)",
    "flag_concentrated_benes": "Concentrated beneficiaries (pill mill pattern)",
    "flag_opioid_heavy":      "High opioid share (>20% of claims)",
    "brand_share":            "Brand drug share",
    "opioid_share":           "Opioid prescription share",
    "cost_per_claim":         "Cost per claim (USD)",
    "claims_per_bene":        "Claims per beneficiary",
}
