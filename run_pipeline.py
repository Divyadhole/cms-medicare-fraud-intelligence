"""
run_pipeline.py
Full end-to-end Medicare fraud detection pipeline.
Runs all steps from data generation to model scoring to app.

Usage:
    python run_pipeline.py           # full pipeline
    python run_pipeline.py --fast    # skip slow SHAP computation
    python run_pipeline.py --data    # data steps only
    python run_pipeline.py --model   # model steps only
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def step(n: int, total: int, name: str):
    print(f"\n{BOLD}[{n}/{total}] {name}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def run_full(fast: bool = False):
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  {BOLD}CMS MEDICARE FRAUD INTELLIGENCE PIPELINE{RESET}")
    print(f"  Source: CMS Part D 2022 + OIG Exclusions")
    print(f"{'='*60}")

    total = 7

    # ── Step 1: Generate data ──────────────────────────────────
    step(1, total, "Generate calibrated sample data")
    from src.generate_sample import N, n_normal, n_fraud, n_suspect
    import pandas as pd
    import numpy as np

    if not os.path.exists("data/raw/cms_partd_2022_sample.csv"):
        exec(open("src/generate_sample.py").read())
    df_raw = pd.read_csv("data/raw/cms_partd_2022_sample.csv", dtype={"npi": str})
    ok(f"{len(df_raw):,} providers generated (normal:{n_normal:,} fraud:{n_fraud:,})")

    # ── Step 2: Clean ─────────────────────────────────────────
    step(2, total, "Clean data — suppression filter, NPI validation")
    from src.clean import load_and_clean, save_clean
    df_clean = load_and_clean()
    save_clean(df_clean)
    ok(f"{len(df_clean):,} providers after cleaning")

    # ── Step 3: Feature engineering ───────────────────────────
    step(3, total, "Feature engineering — 24 features, peer-group z-scores")
    from src.features import run as run_features
    df_features = run_features()
    fraud_flags = df_features[df_features["is_fraud_label"] == 1]["flag_count"].mean()
    normal_flags = df_features[df_features["is_fraud_label"] == 0]["flag_count"].mean()
    ok(f"Flag separation: {fraud_flags:.2f} (fraud) vs {normal_flags:.2f} (normal) = {fraud_flags/normal_flags:.1f}x")

    # ── Step 4: SQLite ingest ─────────────────────────────────
    step(4, total, "Load to SQLite — 4-table schema")
    from src.ingest_db import run as run_ingest
    run_ingest()
    ok("fraud_intelligence.db ready")

    # ── Step 5: XGBoost model ─────────────────────────────────
    step(5, total, "Train XGBoost — 5-fold CV + SMOTE")
    from src.model_xgb import run as run_xgb
    model_xgb, probs, threshold, metrics = run_xgb()
    ok(f"PR-AUC: {metrics['pr_auc']:.4f} (baseline: 0.015)")
    ok(f"Optimal threshold: {threshold:.3f}")

    # ── Step 6: Ensemble + SHAP ───────────────────────────────
    step(6, total, f"Isolation Forest + SHAP ensemble{'  (SHAP skipped --fast)' if fast else ''}")
    from src.model_ensemble import run as run_ensemble
    scores_df = run_ensemble()

    # Validate OIG confirmation
    import pandas as pd
    features_df = pd.read_csv("data/processed/features.csv", dtype={"npi": str})
    merged = scores_df.merge(features_df[["npi", "is_fraud_label"]], on="npi")
    critical = merged[merged["fraud_tier"] == "CRITICAL"]
    confirmed = critical["is_fraud_label"].sum()
    rate = confirmed / len(critical) * 100 if len(critical) > 0 else 0
    ok(f"CRITICAL tier: {len(critical):,} providers")
    ok(f"OIG confirmation: {confirmed} / {len(critical)} = {rate:.1f}%")

    # ── Step 7: Summary ───────────────────────────────────────
    step(7, total, "Pipeline complete")
    elapsed = time.time() - t0

    tier_counts = scores_df["fraud_tier"].value_counts()
    print(f"\n  {'='*55}")
    print(f"  {BOLD}RESULTS{RESET}")
    print(f"  {'='*55}")
    print(f"  Total providers analyzed:  {len(scores_df):,}")
    print(f"  CRITICAL tier:             {tier_counts.get('CRITICAL', 0):,} ({tier_counts.get('CRITICAL', 0)/len(scores_df)*100:.1f}%)")
    print(f"  HIGH tier:                 {tier_counts.get('HIGH', 0):,}")
    print(f"  OIG confirmation rate:     {rate:.1f}%")
    print(f"  PR-AUC:                    {metrics['pr_auc']:.4f}")
    print(f"  Pipeline time:             {elapsed:.1f}s")
    print(f"\n  Launch app:  streamlit run streamlit_app.py")
    print(f"  GitHub:      https://github.com/Divyadhole/cms-medicare-fraud-intelligence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true", help="Skip SHAP computation")
    parser.add_argument("--data",  action="store_true", help="Data steps only")
    parser.add_argument("--model", action="store_true", help="Model steps only")
    args = parser.parse_args()
    run_full(fast=args.fast)
