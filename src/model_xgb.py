"""
src/model_xgb.py
XGBoost supervised fraud classifier.

Key design decisions:
1. SMOTE oversampling — fraud is only 1.5% of data. Without balancing,
   the model learns to predict "not fraud" for everything and gets 98.5%
   accuracy while being completely useless.

2. Precision-Recall AUC as primary metric — not ROC-AUC.
   With 1.5% fraud prevalence, ROC-AUC looks great even for bad models.
   PR-AUC is honest. A random classifier gets PR-AUC = 0.015 (the base rate).

3. Threshold tuning — default 0.5 is wrong for fraud detection.
   We tune to maximize F2-score (recall-weighted) because missing real
   fraud (false negative) is worse than a false alarm (false positive).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, f1_score,
                              classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

FEAT   = Path("data/processed/features.csv")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

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


def load_data():
    df = pd.read_csv(FEAT, dtype={"npi": str})
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["is_fraud_label"].astype(int)
    return df["npi"], X, y, available


def train(X, y):
    print(f"\n  Class distribution:")
    print(f"    Normal: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"    Fraud:  {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # SMOTE: oversample minority class in training folds only
    smote = SMOTE(random_state=42, k_neighbors=5)

    # XGBoost — scale_pos_weight as backup for any remaining imbalance
    fraud_ratio = (y==0).sum() / (y==1).sum()
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_ratio,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print(f"\n  Running 5-fold cross-validation with SMOTE...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_probs = np.zeros(len(y))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE only on training fold
        X_sm, y_sm = smote.fit_resample(X_tr, y_tr)

        model.fit(X_sm, y_sm,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        probs = model.predict_proba(X_val)[:, 1]
        all_probs[val_idx] = probs

        prauc = average_precision_score(y_val, probs)
        rocauc = roc_auc_score(y_val, probs)
        fold_scores.append({"fold": fold+1, "pr_auc": prauc, "roc_auc": rocauc})
        print(f"    Fold {fold+1}: PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}")

    # Final scores
    pr_auc  = average_precision_score(y, all_probs)
    roc_auc = roc_auc_score(y, all_probs)

    print(f"\n  CV Results:")
    print(f"    PR-AUC (primary):  {pr_auc:.4f}")
    print(f"    ROC-AUC:           {roc_auc:.4f}")
    print(f"    Baseline PR-AUC:   {y.mean():.4f} (random classifier)")

    # Threshold tuning: maximize F2-score (recall-weighted)
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_thresh_idx = np.argmax(f2_scores)
    best_threshold  = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5

    y_pred = (all_probs >= best_threshold).astype(int)
    print(f"\n  At optimal threshold ({best_threshold:.3f}):")
    print(classification_report(y, y_pred, target_names=["Normal","Fraud"]))

    # Train final model on all data with SMOTE
    print(f"  Training final model on full dataset...")
    X_final, y_final = smote.fit_resample(X, y)
    model.fit(X_final, y_final, verbose=False)

    return model, all_probs, best_threshold, {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": best_threshold,
        "fold_scores": fold_scores,
    }


def save(model, threshold, metrics, feature_names):
    joblib.dump(model,    MODELS / "xgb_fraud.pkl")
    joblib.dump(threshold, MODELS / "xgb_threshold.pkl")
    joblib.dump(feature_names, MODELS / "feature_names.pkl")

    import json
    clean_metrics = {k: v for k, v in metrics.items() if k != "fold_scores"}
    clean_metrics["fold_scores"] = [
        {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
         for kk, vv in fs.items()}
        for fs in metrics["fold_scores"]
    ]
    with open(MODELS / "xgb_metrics.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in clean_metrics.items()
                   if k != "fold_scores"}, f, indent=2)
    print(f"\n  Model saved → {MODELS}/xgb_fraud.pkl")


def run():
    print("=" * 55)
    print("  XGBOOST FRAUD CLASSIFIER")
    print("=" * 55)

    print("\n[1/3] Loading features...")
    npis, X, y, feature_names = load_data()
    print(f"  {len(X):,} providers, {len(feature_names)} features")

    print("\n[2/3] Training with 5-fold CV + SMOTE...")
    model, probs, threshold, metrics = train(X, y)

    print("\n[3/3] Saving model...")
    save(model, threshold, metrics, feature_names)

    # Save scores back for ensemble
    scores_df = pd.DataFrame({"npi": npis, "fraud_score_xgb": probs})
    scores_df.to_csv("data/processed/xgb_scores.csv", index=False)
    print(f"  Scores saved → data/processed/xgb_scores.csv")

    return model, probs, threshold, metrics


if __name__ == "__main__":
    run()
