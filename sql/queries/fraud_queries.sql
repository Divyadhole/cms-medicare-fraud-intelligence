-- ============================================================
-- sql/queries/fraud_queries.sql
-- Core SQL queries for Medicare fraud investigation
-- ============================================================

-- 1. PEER GROUP OUTLIERS
-- Find providers whose claim volume is extreme vs specialty peers
-- This is how real fraud investigators start — before any ML model
SELECT
    p.npi,
    p.first_name || ' ' || p.last_name AS provider_name,
    p.state,
    p.specialty,
    p.total_claim_count,
    p.total_drug_cost,
    p.cost_per_claim,
    ROUND(AVG(p.total_claim_count) OVER (PARTITION BY p.specialty), 0) AS specialty_avg_claims,
    ROUND(
        (p.total_claim_count - AVG(p.total_claim_count) OVER (PARTITION BY p.specialty))
        / NULLIF(STDDEV(p.total_claim_count) OVER (PARTITION BY p.specialty), 0),
    2) AS z_score_claims,
    ROUND(
        (p.cost_per_claim - AVG(p.cost_per_claim) OVER (PARTITION BY p.specialty))
        / NULLIF(STDDEV(p.cost_per_claim) OVER (PARTITION BY p.specialty), 0),
    2) AS z_score_cost_per_claim
FROM providers p
WHERE z_score_claims > 3.0   -- more than 3 std devs above specialty mean
ORDER BY z_score_claims DESC
LIMIT 50;


-- 2. BRAND-HEAVY PRESCRIBERS
-- Fraud signal: prescribing brand-name drugs when generics are standard
-- Legitimate providers average ~30% brand. Fraud mills often exceed 80%.
SELECT
    p.npi,
    p.first_name || ' ' || p.last_name AS provider_name,
    p.specialty,
    p.state,
    p.brand_share,
    p.total_drug_cost,
    p.total_claim_count,
    ROUND(AVG(p.brand_share) OVER (PARTITION BY p.specialty), 3) AS specialty_avg_brand_share,
    ROUND(p.brand_share - AVG(p.brand_share) OVER (PARTITION BY p.specialty), 3)
        AS brand_share_above_avg
FROM providers p
WHERE p.brand_share > 0.75
  AND p.total_claim_count > 500
ORDER BY p.brand_share DESC
LIMIT 50;


-- 3. CONCENTRATED BENEFICIARIES — PILL MILL PATTERN
-- Fraud mills process huge volume on few patients
-- Normal: ~5-8 claims per beneficiary. Pill mills: 20-50+
SELECT
    p.npi,
    p.first_name || ' ' || p.last_name AS provider_name,
    p.specialty,
    p.state,
    p.total_claim_count,
    p.total_beneficiaries,
    p.claims_per_bene,
    ROUND(AVG(p.claims_per_bene) OVER (PARTITION BY p.specialty), 2) AS specialty_avg_cpb,
    RANK() OVER (PARTITION BY p.specialty ORDER BY p.claims_per_bene DESC)
        AS rank_within_specialty
FROM providers p
WHERE p.total_claim_count > 200
ORDER BY p.claims_per_bene DESC
LIMIT 50;


-- 4. YEAR-OVER-YEAR VELOCITY (requires multi-year data)
-- Sudden surge in prescription volume is a major fraud signal
-- Normal growth: 5-15% per year. Fraud: 200-500% overnight.
WITH yoy AS (
    SELECT npi, specialty, data_year, total_claim_count,
        LAG(total_claim_count) OVER (PARTITION BY npi ORDER BY data_year)
            AS prior_year_claims
    FROM providers
)
SELECT
    npi, specialty, data_year,
    total_claim_count,
    prior_year_claims,
    ROUND(100.0 * (total_claim_count - prior_year_claims)
        / NULLIF(prior_year_claims, 0), 1) AS yoy_growth_pct
FROM yoy
WHERE prior_year_claims IS NOT NULL
  AND (total_claim_count - prior_year_claims) / NULLIF(prior_year_claims, 0) > 2.0
ORDER BY yoy_growth_pct DESC;


-- 5. OPIOID HIGH-PRESCRIBERS BY STATE
-- Identify opioid outliers within each state
SELECT
    p.state,
    p.specialty,
    p.npi,
    p.first_name || ' ' || p.last_name AS provider_name,
    p.opioid_share,
    p.opioid_claim_count,
    p.total_claim_count,
    RANK() OVER (PARTITION BY p.state ORDER BY p.opioid_share DESC)
        AS state_opioid_rank,
    ROUND(AVG(p.opioid_share) OVER (PARTITION BY p.state), 3)
        AS state_avg_opioid_share
FROM providers p
WHERE p.total_claim_count > 100
  AND p.opioid_share > 0.20
ORDER BY p.state, p.opioid_share DESC;


-- 6. MULTI-FLAG PROVIDERS — HIGHEST PRIORITY
-- Providers flagged on multiple dimensions simultaneously
SELECT
    p.npi,
    p.first_name || ' ' || p.last_name AS provider_name,
    p.specialty,
    p.state,
    p.total_drug_cost,
    f.flag_count,
    f.flag_high_volume,
    f.flag_high_cost_per_claim,
    f.flag_brand_heavy,
    f.flag_concentrated_benes,
    f.flag_opioid_heavy,
    p.fraud_score_ensemble,
    p.fraud_tier
FROM providers p
JOIN provider_features f ON p.npi = f.npi
WHERE f.flag_count >= 3    -- hit 3 or more independent fraud signals
ORDER BY f.flag_count DESC, p.fraud_score_ensemble DESC
LIMIT 100;


-- 7. STATE FRAUD CONCENTRATION MAP
-- Which states have highest concentration of high-risk providers?
SELECT
    p.state,
    COUNT(*) AS total_providers,
    SUM(CASE WHEN p.fraud_tier IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) AS high_risk_count,
    ROUND(100.0 * SUM(CASE WHEN p.fraud_tier IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END)
        / COUNT(*), 2) AS high_risk_pct,
    ROUND(SUM(p.total_drug_cost) / 1e6, 1) AS total_cost_M,
    ROUND(AVG(p.fraud_score_ensemble), 3) AS avg_fraud_score
FROM providers p
WHERE p.fraud_tier IS NOT NULL
GROUP BY p.state
ORDER BY high_risk_pct DESC;


-- 8. OIG CROSS-REFERENCE — CONFIRMED CATCHES
-- How many model-flagged providers are already on OIG exclusions list?
SELECT
    p.fraud_tier,
    COUNT(*) AS total_in_tier,
    SUM(CASE WHEN o.npi IS NOT NULL THEN 1 ELSE 0 END) AS oig_confirmed,
    ROUND(100.0 * SUM(CASE WHEN o.npi IS NOT NULL THEN 1 ELSE 0 END)
        / COUNT(*), 2) AS oig_match_pct
FROM providers p
LEFT JOIN oig_exclusions o ON p.npi = o.npi
GROUP BY p.fraud_tier
ORDER BY
    CASE p.fraud_tier
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH'     THEN 2
        WHEN 'MODERATE' THEN 3
        ELSE 4
    END;
