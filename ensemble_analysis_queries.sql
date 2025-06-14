-- ENSEMBLE STRATEGY ANALYSIS QUERIES
-- Run these queries in sql_analytics.py interactive mode
-- ============================================================

-- SIGNAL TIMING VALIDATION
-- ------------------------------------------------------------

-- Check for look-ahead bias in signal timing
SELECT 
    'Signal Timing Check' as analysis,
    COUNT(CASE WHEN entry_bar < signal_bar THEN 1 END) as lookahead_violations,
    COUNT(*) as total_trades,
    ROUND(100.0 * COUNT(CASE WHEN entry_bar < signal_bar THEN 1 END) / COUNT(*), 2) as violation_pct
FROM signal_performance
WHERE entry_bar IS NOT NULL;


-- STRATEGY CORRELATION MATRIX
-- ------------------------------------------------------------

-- Find low-correlation strategy pairs for ensemble
WITH strategy_pairs AS (
    SELECT DISTINCT
        s1.component_id as strategy_a,
        s2.component_id as strategy_b,
        s1.strategy_type as type_a,
        s2.strategy_type as type_b
    FROM component_metrics s1
    CROSS JOIN component_metrics s2
    WHERE s1.component_type = 'strategy' 
    AND s2.component_type = 'strategy'
    AND s1.component_id < s2.component_id
)
SELECT 
    type_a || ' + ' || type_b as strategy_combination,
    COUNT(*) as pair_count,
    'Low correlation pairs ideal for ensemble' as note
FROM strategy_pairs
WHERE type_a != type_b
GROUP BY type_a, type_b
ORDER BY pair_count DESC;


-- COMPLEMENTARY SIGNALS
-- ------------------------------------------------------------

-- Find strategies with complementary signal patterns
WITH signal_summary AS (
    SELECT 
        component_id,
        COUNT(DISTINCT bar_index) as signal_bars,
        MIN(bar_index) as first_signal,
        MAX(bar_index) as last_signal,
        COUNT(*) as total_signals
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY component_id
),
overlaps AS (
    SELECT 
        s1.component_id as strategy_a,
        s2.component_id as strategy_b,
        s1.signal_bars as bars_a,
        s2.signal_bars as bars_b,
        COUNT(DISTINCT sc1.bar_index) as overlapping_bars
    FROM signal_summary s1
    CROSS JOIN signal_summary s2
    LEFT JOIN signal_changes sc1 ON sc1.component_id = s1.component_id
    LEFT JOIN signal_changes sc2 ON sc2.component_id = s2.component_id 
        AND sc2.bar_index = sc1.bar_index
    WHERE s1.component_id < s2.component_id
    AND s1.signal_bars > 10 
    AND s2.signal_bars > 10
    GROUP BY s1.component_id, s2.component_id, s1.signal_bars, s2.signal_bars
)
SELECT 
    strategy_a,
    strategy_b,
    bars_a + bars_b - overlapping_bars as combined_coverage,
    ROUND(100.0 * overlapping_bars / LEAST(bars_a, bars_b), 2) as overlap_pct,
    CASE 
        WHEN overlap_pct < 30 THEN 'Excellent diversity'
        WHEN overlap_pct < 50 THEN 'Good diversity'
        ELSE 'Consider alternatives'
    END as ensemble_quality
FROM overlaps
WHERE overlap_pct < 50
ORDER BY combined_coverage DESC
LIMIT 20;


-- REGIME BASED PERFORMANCE
-- ------------------------------------------------------------

-- Analyze strategy performance by market regime
WITH regime_signals AS (
    SELECT 
        sc.bar_index,
        sc.component_id as classifier_id,
        sc.signal_value as regime,
        ss.component_id as strategy_id,
        ss.signal_value as strategy_signal
    FROM signal_changes sc
    JOIN signal_changes ss ON ss.bar_index = sc.bar_index
    WHERE sc.component_type = 'classifier'
    AND ss.component_type = 'strategy'
)
SELECT 
    classifier_id,
    regime,
    strategy_id,
    COUNT(*) as signals_in_regime,
    COUNT(DISTINCT bar_index) as unique_bars
FROM regime_signals
GROUP BY classifier_id, regime, strategy_id
HAVING signals_in_regime > 5
ORDER BY classifier_id, regime, signals_in_regime DESC;


-- OPTIMAL ENSEMBLE SIZE
-- ------------------------------------------------------------

-- Determine optimal number of strategies in ensemble
WITH strategy_counts AS (
    SELECT 
        bar_index,
        COUNT(DISTINCT component_id) as strategies_signaling,
        COUNT(DISTINCT signal_value) as unique_signals,
        CASE 
            WHEN COUNT(DISTINCT signal_value) = 1 THEN 'Unanimous'
            WHEN COUNT(DISTINCT signal_value) = 2 THEN 'Mixed'
            ELSE 'Divergent'
        END as signal_agreement
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY bar_index
)
SELECT 
    strategies_signaling,
    signal_agreement,
    COUNT(*) as occurrence_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct_of_bars
FROM strategy_counts
GROUP BY strategies_signaling, signal_agreement
ORDER BY strategies_signaling, signal_agreement;


-- ENSEMBLE VOTING ANALYSIS
-- ------------------------------------------------------------

-- Analyze potential voting outcomes
WITH signal_votes AS (
    SELECT 
        bar_index,
        SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as long_votes,
        SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as short_votes,
        SUM(CASE WHEN signal_value = 0 THEN 1 ELSE 0 END) as neutral_votes,
        COUNT(*) as total_votes
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY bar_index
)
SELECT 
    CASE 
        WHEN long_votes > short_votes AND long_votes > neutral_votes THEN 'Long Consensus'
        WHEN short_votes > long_votes AND short_votes > neutral_votes THEN 'Short Consensus'
        WHEN neutral_votes >= total_votes / 2 THEN 'Neutral Consensus'
        ELSE 'No Clear Consensus'
    END as voting_outcome,
    COUNT(*) as bars_count,
    AVG(total_votes) as avg_strategies_voting,
    MAX(long_votes) as max_long_votes,
    MAX(short_votes) as max_short_votes
FROM signal_votes
GROUP BY voting_outcome
ORDER BY bars_count DESC;


-- PARAMETER SENSITIVITY
-- ------------------------------------------------------------

-- Analyze parameter sensitivity for ensemble tuning
SELECT 
    strategy_type,
    COUNT(DISTINCT component_id) as variations,
    MIN(signal_frequency) as min_signal_freq,
    AVG(signal_frequency) as avg_signal_freq,
    MAX(signal_frequency) as max_signal_freq,
    STDDEV(signal_frequency) as signal_freq_stddev,
    CASE 
        WHEN STDDEV(signal_frequency) < 0.01 THEN 'Low sensitivity'
        WHEN STDDEV(signal_frequency) < 0.05 THEN 'Medium sensitivity'
        ELSE 'High sensitivity'
    END as parameter_sensitivity
FROM component_metrics
WHERE component_type = 'strategy'
GROUP BY strategy_type
HAVING COUNT(DISTINCT component_id) > 3
ORDER BY avg_signal_freq DESC;


