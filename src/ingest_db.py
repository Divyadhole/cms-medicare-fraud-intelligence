"""
src/ingest_db.py
Loads all processed data into SQLite for SQL-based fraud investigation.
Creates all 5 tables, runs validation queries after load.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB   = Path("data/fraud_intelligence.db")
DB.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = Path("sql/schema/create_tables.sql")
CLEAN  = Path("data/processed/cms_cleaned.csv")
FEAT   = Path("data/processed/features.csv")
BENCH  = Path("data/processed/specialty_benchmarks.csv")


def init_db(conn: sqlite3.Connection) -> None:
    with open(SCHEMA) as f:
        conn.executescript(f.read())
    conn.commit()
    print("  Schema created")


def load_providers(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(CLEAN, dtype={"npi": str})
    df = df.rename(columns={
        "nppes_provider_last_org_name": "last_name",
        "nppes_provider_first_name":    "first_name",
        "nppes_provider_state":         "state",
        "specialty_description":        "specialty",
    })
    df["data_year"] = 2022
    df.to_sql("providers", conn, if_exists="replace", index=False)
    conn.commit()
    return len(df)


def load_features(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(FEAT, dtype={"npi": str})
    df = df.rename(columns={"specialty_description": "specialty"})

    flag_cols = ["flag_high_volume","flag_high_cost_per_claim",
                 "flag_brand_heavy","flag_concentrated_benes",
                 "flag_opioid_heavy","flag_count"]
    feat_cols = ["npi","specialty"] + \
                [c for c in df.columns if c.startswith(("z_","pct_","flag_"))]
    sub = df[[c for c in feat_cols if c in df.columns]]
    sub.to_sql("provider_features", conn, if_exists="replace", index=False)
    conn.commit()
    return len(sub)


def load_benchmarks(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(BENCH)
    df.to_sql("specialty_benchmarks", conn, if_exists="replace", index=False)
    conn.commit()
    return len(df)


def validate(conn: sqlite3.Connection) -> None:
    checks = [
        ("Total providers",           "SELECT COUNT(*) FROM providers"),
        ("Fraud labels",              "SELECT COUNT(*) FROM providers WHERE is_fraud_label=1"),
        ("Specialties",               "SELECT COUNT(DISTINCT specialty) FROM providers"),
        ("States",                    "SELECT COUNT(DISTINCT state) FROM providers"),
        ("Feature rows",              "SELECT COUNT(*) FROM provider_features"),
        ("Providers 3+ flags",        "SELECT COUNT(*) FROM provider_features WHERE flag_count>=3"),
        
    ]
    print("\n  Validation:")
    for label, sql in checks:
        val = conn.execute(sql).fetchone()[0]
        print(f"    {label:<30} {val:,}")

    # Top 5 specialties by total cost
    print("\n  Top 5 specialties by total drug cost:")
    rows = conn.execute("""
        SELECT specialty, COUNT(*) n,
               ROUND(SUM(total_drug_cost)/1e6,1) cost_M
        FROM providers GROUP BY specialty
        ORDER BY cost_M DESC LIMIT 5
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:<25} {r[1]:,} providers  ${r[2]}M")


def run():
    print("=" * 55)
    print("  SQLITE INGEST PIPELINE")
    print("=" * 55)

    if DB.exists():
        DB.unlink()
        print(f"\n  Existing DB removed")

    conn = sqlite3.connect(DB)
    print(f"\n[1/4] Initializing schema...")
    init_db(conn)

    print(f"\n[2/4] Loading providers...")
    n = load_providers(conn)
    print(f"  {n:,} rows loaded")

    print(f"\n[3/4] Loading features...")
    n = load_features(conn)
    print(f"  {n:,} rows loaded")

    print(f"\n[4/4] Loading benchmarks...")
    n = load_benchmarks(conn)
    print(f"  {n} specialties loaded")

    validate(conn)
    conn.close()

    size_mb = DB.stat().st_size / 1e6
    print(f"\n  DB saved → {DB}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    run()
