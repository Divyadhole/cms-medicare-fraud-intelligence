"""
src/model_ensemble.py
Isolation Forest (unsupervised) + SHAP explainability + ensemble scoring.

Why two models:
  XGBoost supervised:   catches known fraud patterns (OIG-labeled)
  Isolation Forest:     catches novel patterns with no labels yet

The most dangerous fraud is the kind that hasn't been caught before.
Isolation Forest finds statistical outliers regardless of label history.

Ensemble: weighted average (XGB 60% + IsoForest 40%)
Providers flagged by both get highest priority.
"""

import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

FEAT    = Path("data/processed/features.csv")
XGB_SCR = Path("data/processed/xgb_scores.csv")
MODELS  = Path("models")
DB      = Path("data/fraud_intelligence.db")

FEATURE_COLS = [
    "total_claim_count", "total_drug_cost", "total_beneficiaries",
    "cost_per_claim", "cost_per_beneficiary", "brand_share",
    "opioid_share", "claims_per_bene",
    "z_total_claim_count", "z_total_drug_cost", "z_cost_per_claim",
    "z_brand_share", "z_opioid_share", "z_claims_per_bene",
    "pct_total_claim_count", "pct_total_drug_cost",
    "pct_cost_per_claim", "pct_brand_share",
    "flag_high_volume", "flag_high_cost_per_claim",
    "flag_brand_heavy", "flag_concentrated_benes",
    "flag_opioid_heavy", "flag_count",
]


def run_isolation_forest(X: pd.DataFrame) -> np.ndarray:
    """
    Isolation Forest: anomaly score per provider.
    contamination=0.03 means we expect ~3% anomalies in data.
    Returns normalized 0-1 score (higher = more anomalous).
    """
    print("  Training Isolation Forest (unsupervised)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)

    # decision_function: negative = anomalous, positive = normal
    raw_scores = iso.decision_function(X)

    # Flip and normalize to 0-1 (higher = more anomalous)
    flipped = -raw_scores
    scaler  = MinMaxScaler()
    normalized = scaler.fit_transform(flipped.reshape(-1,1)).flatten()

    joblib.dump(iso,    MODELS / "iso_forest.pkl")
    joblib.dump(scaler, MODELS / "iso_scaler.pkl")
    print(f"  Isolation Forest saved → {MODELS}/iso_forest.pkl")
    return normalized


def compute_ensemble(xgb_probs: np.ndarray,
                     iso_scores: np.ndarray,
                     w_xgb: float = 0.60,
                     w_iso: float = 0.40) -> np.ndarray:
    return (w_xgb * xgb_probs + w_iso * iso_scores)


def assign_tiers(scores: np.ndarray) -> list:
    tiers = []
    for s in scores:
        if   s >= 0.80: tiers.append("CRITICAL")
        elif s >= 0.60: tiers.append("HIGH")
        elif s >= 0.40: tiers.append("MODERATE")
        else:           tiers.append("LOW")
    return tiers


def compute_shap(model, X: pd.DataFrame,
                 feature_names: list,
                 n_background: int = 500) -> pd.DataFrame:
    """Compute SHAP values for top 1000 highest-risk providers."""
    print("  Computing SHAP values for top-risk providers...")
    background = shap.maskers.Independent(X.sample(n_background, random_state=42))
    explainer  = shap.Explainer(model, background)
    shap_vals  = explainer(X, check_additivity=False)

    # Build top-3 reasons per provider
    reasons = []
    for i in range(len(X)):
        vals = shap_vals.values[i]
        top3 = np.argsort(np.abs(vals))[::-1][:3]
        reasons.append({
            "reason_1": f"{feature_names[top3[0]]}: {X.iloc[i, top3[0]]:.2f} (impact {vals[top3[0]]:+.3f})",
            "reason_2": f"{feature_names[top3[1]]}: {X.iloc[i, top3[1]]:.2f} (impact {vals[top3[1]]:+.3f})",
            "reason_3": f"{feature_names[top3[2]]}: {X.iloc[i, top3[2]]:.2f} (impact {vals[top3[2]]:+.3f})",
        })
    return pd.DataFrame(reasons)


def run():
    print("=" * 55)
    print("  ENSEMBLE SCORING + SHAP EXPLAINABILITY")
    print("=" * 55)

    print("\n[1/5] Loading data and XGB scores...")
    df      = pd.read_csv(FEAT, dtype={"npi": str})
    xgb_scr = pd.read_csv(XGB_SCR, dtype={"npi": str})
    model   = joblib.load(MODELS / "xgb_fraud.pkl")
    feat_names = joblib.load(MODELS / "feature_names.pkl")

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)

    print("\n[2/5] Running Isolation Forest...")
    iso_scores = run_isolation_forest(X)

    xgb_probs  = xgb_scr.set_index("npi").reindex(df["npi"])["fraud_score_xgb"].values

    print("\n[3/5] Computing ensemble scores...")
    ensemble   = compute_ensemble(xgb_probs, iso_scores)
    tiers      = assign_tiers(ensemble)

    results = pd.DataFrame({
        "npi":                 df["npi"],
        "fraud_score_xgb":    xgb_probs,
        "fraud_score_iso":    iso_scores,
        "fraud_score_ensemble": ensemble,
        "fraud_tier":          tiers,
    })

    tier_counts = results["fraud_tier"].value_counts()
    print(f"\n  Tier distribution:")
    for tier in ["CRITICAL","HIGH","MODERATE","LOW"]:
        n = tier_counts.get(tier, 0)
        print(f"    {tier:<10} {n:,} providers ({n/len(results)*100:.1f}%)")

    print("\n[4/5] Computing SHAP explanations for top 500 providers...")
    top500 = results.nlargest(500, "fraud_score_ensemble")
    top_idx = df[df["npi"].isin(top500["npi"])].index
    X_top  = X.loc[top_idx]

    shap_df = compute_shap(model, X_top, feat_names)
    shap_df.insert(0, "npi", df.loc[top_idx, "npi"].values)

    print("\n[5/5] Saving all outputs...")
    results.to_csv("data/processed/ensemble_scores.csv", index=False)
    shap_df.to_csv("data/processed/shap_reasons.csv",   index=False)

    # Check OIG cross-match (using is_fraud_label as proxy)
    if "is_fraud_label" in df.columns:
        merged = results.merge(df[["npi","is_fraud_label"]], on="npi")
        critical = merged[merged["fraud_tier"]=="CRITICAL"]
        confirmed = critical["is_fraud_label"].sum()
        print(f"\n  OIG Cross-Reference:")
        print(f"    CRITICAL tier providers:   {len(critical):,}")
        print(f"    Already flagged (OIG):     {confirmed}")
        print(f"    Confirmation rate:         {confirmed/len(critical)*100:.1f}%")

    print(f"\n  Saved → data/processed/ensemble_scores.csv")
    print(f"  Saved → data/processed/shap_reasons.csv")
    return results


if __name__ == "__main__":
    run()
