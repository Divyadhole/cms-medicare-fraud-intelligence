-- ============================================================
-- sql/schema/create_tables.sql
-- Medicare Part D Fraud Intelligence Database Schema
-- ============================================================

-- Core provider table (one row per NPI)
CREATE TABLE IF NOT EXISTS providers (
    npi                       TEXT PRIMARY KEY,
    last_name                 TEXT,
    first_name                TEXT,
    state                     TEXT,
    specialty                 TEXT,
    total_claim_count         INTEGER,
    total_30day_fill_count    INTEGER,
    total_drug_cost           REAL,
    total_beneficiaries       INTEGER,
    brand_claim_count         INTEGER,
    generic_claim_count       INTEGER,
    opioid_claim_count        INTEGER,
    opioid_bene_count         INTEGER,
    -- Derived
    cost_per_claim            REAL,
    cost_per_beneficiary      REAL,
    brand_share               REAL,
    opioid_share              REAL,
    claims_per_bene           REAL,
    -- Ground truth (OIG cross-match)
    is_fraud_label            INTEGER DEFAULT 0,
    -- Fraud scores (populated after modeling)
    fraud_score_xgb           REAL,
    fraud_score_iso           REAL,
    fraud_score_ensemble      REAL,
    fraud_tier                TEXT,  -- LOW / MODERATE / HIGH / CRITICAL
    -- Metadata
    data_year                 INTEGER DEFAULT 2022,
    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Peer group benchmarks (one row per specialty)
CREATE TABLE IF NOT EXISTS specialty_benchmarks (
    specialty                 TEXT PRIMARY KEY,
    n_providers               INTEGER,
    avg_claim_count           REAL,
    std_claim_count           REAL,
    avg_drug_cost             REAL,
    std_drug_cost             REAL,
    avg_cost_per_claim        REAL,
    std_cost_per_claim        REAL,
    avg_brand_share           REAL,
    std_brand_share           REAL,
    avg_opioid_share          REAL,
    std_opioid_share          REAL,
    p95_claim_count           REAL,
    p99_claim_count           REAL,
    p99_cost_per_claim        REAL
);

-- Provider features (engineered, one row per NPI)
CREATE TABLE IF NOT EXISTS provider_features (
    npi                       TEXT PRIMARY KEY,
    specialty                 TEXT,
    -- Z-scores vs specialty peer group
    z_claim_count             REAL,
    z_drug_cost               REAL,
    z_cost_per_claim          REAL,
    z_brand_share             REAL,
    z_opioid_share            REAL,
    z_claims_per_bene         REAL,
    -- Percentile ranks within specialty
    pct_rank_claims           REAL,
    pct_rank_cost             REAL,
    pct_rank_cost_per_claim   REAL,
    -- Binary flags
    flag_high_volume          INTEGER,  -- > 99th pct claims in specialty
    flag_high_cost_per_claim  INTEGER,  -- > 99th pct cost/claim in specialty
    flag_brand_heavy          INTEGER,  -- brand_share > 0.70
    flag_concentrated_benes   INTEGER,  -- claims_per_bene > specialty p95
    flag_opioid_heavy         INTEGER,  -- opioid_share > 0.20
    -- Composite
    flag_count                INTEGER,  -- total flags (0-5)
    FOREIGN KEY (npi) REFERENCES providers(npi)
);

-- OIG exclusions cross-reference
CREATE TABLE IF NOT EXISTS oig_exclusions (
    npi                       TEXT,
    exclusion_date            TEXT,
    reinstatement_date        TEXT,
    exclusion_type            TEXT,
    general                   TEXT,     -- reason category
    PRIMARY KEY (npi, exclusion_date)
);

-- Fraud alerts (model output)
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    npi                       TEXT,
    alert_date                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fraud_score               REAL,
    tier                      TEXT,
    top_reason_1              TEXT,
    top_reason_2              TEXT,
    top_reason_3              TEXT,
    model_version             TEXT DEFAULT 'v1.0',
    FOREIGN KEY (npi) REFERENCES providers(npi)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_providers_state     ON providers(state);
CREATE INDEX IF NOT EXISTS idx_providers_specialty ON providers(specialty);
CREATE INDEX IF NOT EXISTS idx_providers_tier      ON providers(fraud_tier);
CREATE INDEX IF NOT EXISTS idx_features_flags      ON provider_features(flag_count DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_score        ON fraud_alerts(fraud_score DESC);
