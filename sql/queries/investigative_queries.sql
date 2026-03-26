-- ============================================================
-- sql/queries/investigative_queries.sql
-- Additional investigative SQL for fraud analysts
-- Goes beyond the basic fraud_queries.sql
-- ============================================================

-- 1. NETWORK ANALYSIS: providers sharing the same beneficiary pool
-- High overlap in patient base between two providers = coordination signal
WITH bene_provider AS (
    SELECT npi, specialty,
           total_beneficiaries,
           total_claim_count,
           claims_per_bene
    FROM providers
    WHERE total_beneficiaries < 50     -- small patient pool
      AND total_claim_count > 200       -- but very high volume
),
flagged AS (
    SELECT p.npi, p.specialty,
           f.flag_count,
           p.fraud_score_ensemble,
           p.total_beneficiaries,
           p.total_claim_count
    FROM providers p
    JOIN provider_features f ON p.npi = f.npi
    WHERE f.flag_concentrated_benes = 1
      AND f.flag_high_volume = 1
)
SELECT f.*, p.fraud_tier
FROM flagged f
JOIN providers p ON f.npi = p.npi
ORDER BY f.total_claim_count / f.total_beneficiaries DESC
LIMIT 50;


-- 2. SPECIALTY ANOMALY DETECTION
-- Specialties where the 99th percentile looks nothing like the median
-- Huge spread = likely fraud pulling up the top
SELECT specialty,
    COUNT(*) n_providers,
    ROUND(AVG(total_drug_cost), 0)      avg_cost,
    ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY total_drug_cost), 0)
                                         median_cost,
    ROUND(percentile_cont(0.99) WITHIN GROUP (ORDER BY total_drug_cost), 0)
                                         p99_cost,
    ROUND(
        percentile_cont(0.99) WITHIN GROUP (ORDER BY total_drug_cost)
        / NULLIF(percentile_cont(0.50) WITHIN GROUP (ORDER BY total_drug_cost), 0),
    1) p99_to_median_ratio
FROM providers
GROUP BY specialty
HAVING COUNT(*) > 50
ORDER BY p99_to_median_ratio DESC;


-- 3. YEAR-OVER-YEAR VELOCITY (multi-year data required)
-- Providers whose volume surged > 100% in a single year
-- Legitimate practices grow 5-20%/yr. 100%+ surge = red flag.
WITH yoy AS (
    SELECT npi, specialty, data_year,
           total_claim_count AS claims_this_yr,
           LAG(total_claim_count) OVER (PARTITION BY npi ORDER BY data_year)
               AS claims_prior_yr
    FROM providers
)
SELECT npi, specialty, data_year,
    claims_this_yr,
    claims_prior_yr,
    ROUND(100.0 * (claims_this_yr - claims_prior_yr)
        / NULLIF(claims_prior_yr, 0), 1) AS growth_pct
FROM yoy
WHERE claims_prior_yr IS NOT NULL
  AND (claims_this_yr - claims_prior_yr) / NULLIF(claims_prior_yr, 0) > 1.0
ORDER BY growth_pct DESC;


-- 4. GEOGRAPHIC CLUSTERING
-- States with disproportionately high CRITICAL provider density
-- Fraud often clusters geographically (e.g., South Florida, Houston)
SELECT p.state,
    COUNT(*)                                      total_providers,
    SUM(CASE WHEN p.fraud_tier = 'CRITICAL' THEN 1 ELSE 0 END)
                                                   critical_count,
    ROUND(100.0 * SUM(CASE WHEN p.fraud_tier = 'CRITICAL' THEN 1 ELSE 0 END)
        / COUNT(*), 2)                             critical_pct,
    ROUND(AVG(p.fraud_score_ensemble), 4)          avg_ensemble_score,
    ROUND(SUM(p.total_drug_cost) / 1e6, 1)        total_cost_M,
    ROUND(SUM(CASE WHEN p.fraud_tier = 'CRITICAL' THEN p.total_drug_cost ELSE 0 END)
        / 1e6, 1)                                  at_risk_cost_M
FROM providers p
GROUP BY p.state
ORDER BY critical_pct DESC;


-- 5. SHAP REASON ANALYSIS
-- What are the most common primary fraud signals across CRITICAL providers?
-- Helps prioritize which features to focus investigative resources on
SELECT
    CASE
        WHEN reason_1 LIKE '%z_total_claim_count%' THEN 'Extreme claim volume'
        WHEN reason_1 LIKE '%z_cost_per_claim%'    THEN 'High cost per claim'
        WHEN reason_1 LIKE '%z_brand_share%'       THEN 'Brand-heavy prescribing'
        WHEN reason_1 LIKE '%z_opioid_share%'      THEN 'Opioid overuse'
        WHEN reason_1 LIKE '%flag_count%'          THEN 'Multiple flags combined'
        WHEN reason_1 LIKE '%claims_per_bene%'     THEN 'Concentrated beneficiaries'
        ELSE 'Other'
    END AS primary_signal,
    COUNT(*) providers_with_signal,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fraud_alerts), 1) pct_of_alerts
FROM fraud_alerts a
JOIN providers p ON a.npi = p.npi
WHERE p.fraud_tier = 'CRITICAL'
GROUP BY primary_signal
ORDER BY providers_with_signal DESC;


-- 6. FALSE POSITIVE AUDIT
-- CRITICAL providers with only 1-2 flags and no OIG match
-- These are the ones most likely to be legitimate outliers worth reviewing
SELECT p.npi,
    p.specialty, p.state,
    p.fraud_score_ensemble,
    f.flag_count,
    p.total_claim_count,
    p.brand_share,
    p.opioid_share,
    p.is_fraud_label
FROM providers p
JOIN provider_features f ON p.npi = f.npi
WHERE p.fraud_tier = 'CRITICAL'
  AND f.flag_count <= 2
  AND p.is_fraud_label = 0
ORDER BY p.fraud_score_ensemble DESC;
