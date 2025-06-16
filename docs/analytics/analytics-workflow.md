# Systematic Classifier and Strategy Analytics Workflow

This document provides a comprehensive workflow for evaluating ~100 classifier parameter combinations and ~1000 strategy combinations to build an optimal ensemble trading system. The goal is to identify classifiers that actually work and pair them with strategies that perform best under specific regime conditions.

## Prerequisites

- DuckDB installed and configured
- Parquet files with classifier states, signals, and market data
- At least 1 month of 1-minute data for statistical significance
- Paper trading environment ready for validation

## Phase 1: Classifier Validation & Quality Assessment

The first phase focuses on determining which classifiers are actually useful. Many parameter combinations will produce classifiers that either don't meaningfully differentiate market conditions or are too noisy to be actionable.

### Step 1.1: State Distribution Analysis

**Motivation:** A good classifier should have reasonably balanced state distributions. If a classifier spends 95% of its time in one state, it's not providing useful regime differentiation. Conversely, if states change too frequently, the classifier may be overfitting to noise.

```sql
-- Check if classifier states have reasonable distribution
-- Target: ~20% per state for 5-state classifiers, ~33% for 3-state classifiers
SELECT 
    classifier_id, 
    regime, 
    COUNT(*) as occurrences,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) as pct_time,
    -- Flag problematic distributions
    CASE 
        WHEN COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) < 5 THEN 'RARE_STATE'
        WHEN COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) > 60 THEN 'DOMINANT_STATE'
        ELSE 'BALANCED'
    END as distribution_flag
FROM classifier_states 
GROUP BY classifier_id, regime
ORDER BY classifier_id, pct_time DESC;

-- Summary view of classifier balance
WITH distribution_stats AS (
    SELECT 
        classifier_id,
        COUNT(DISTINCT regime) as num_states,
        MAX(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id)) as max_state_pct,
        MIN(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id)) as min_state_pct,
        STDDEV(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id)) as pct_std
    FROM classifier_states 
    GROUP BY classifier_id, regime
)
SELECT 
    classifier_id,
    num_states,
    max_state_pct,
    min_state_pct,
    pct_std,
    -- Quality score: lower std deviation = more balanced
    CASE 
        WHEN pct_std < 5 THEN 'EXCELLENT'
        WHEN pct_std < 10 THEN 'GOOD'
        WHEN pct_std < 20 THEN 'FAIR'
        ELSE 'POOR'
    END as balance_quality
FROM distribution_stats
ORDER BY pct_std ASC;
```

**Red Flags to Watch For:**
- Any state representing <5% of total time (may not have enough data for reliable analysis)
- Any state representing >60% of total time (classifier not differentiating well)
- High standard deviation in state percentages (severely unbalanced)

### Step 1.2: Regime Persistence & Stability Analysis

**Motivation:** Good regime classifiers should produce states that persist for meaningful periods. If regimes change every few minutes, the classifier is likely reacting to noise rather than detecting genuine market structure changes. For 1-minute trading, regimes should ideally last 10-30 minutes to be actionable.

```sql
-- Calculate regime duration statistics
WITH regime_transitions AS (
    SELECT 
        classifier_id,
        regime,
        timestamp,
        LAG(regime) OVER (PARTITION BY classifier_id ORDER BY timestamp) as prev_regime,
        LAG(timestamp) OVER (PARTITION BY classifier_id ORDER BY timestamp) as prev_timestamp
    FROM classifier_states
    WHERE confidence > 0.6  -- Only consider high-confidence classifications
),
regime_durations AS (
    SELECT 
        classifier_id,
        regime,
        timestamp - prev_timestamp as duration_minutes,
        confidence
    FROM regime_transitions rt
    JOIN classifier_states cs ON rt.classifier_id = cs.classifier_id 
        AND rt.timestamp = cs.timestamp
    WHERE prev_regime != regime OR prev_regime IS NULL  -- Regime change points
)
SELECT 
    classifier_id,
    regime,
    COUNT(*) as regime_occurrences,
    AVG(duration_minutes) as avg_duration_min,
    MEDIAN(duration_minutes) as median_duration_min,
    STDDEV(duration_minutes) as duration_std,
    MIN(duration_minutes) as min_duration,
    MAX(duration_minutes) as max_duration,
    -- Stability metrics
    COUNT(CASE WHEN duration_minutes < 5 THEN 1 END) * 100.0 / COUNT(*) as pct_short_regimes,
    COUNT(CASE WHEN duration_minutes > 30 THEN 1 END) * 100.0 / COUNT(*) as pct_long_regimes
FROM regime_durations
GROUP BY classifier_id, regime
HAVING COUNT(*) >= 10  -- Need sufficient regime transitions for analysis
ORDER BY classifier_id, avg_duration_min DESC;

-- Classifier stability summary
WITH stability_metrics AS (
    SELECT 
        classifier_id,
        AVG(avg_duration_min) as overall_avg_duration,
        AVG(pct_short_regimes) as avg_pct_short,
        COUNT(DISTINCT regime) as active_regimes
    FROM regime_duration_analysis
    GROUP BY classifier_id
)
SELECT 
    classifier_id,
    overall_avg_duration,
    avg_pct_short,
    active_regimes,
    CASE 
        WHEN overall_avg_duration > 20 AND avg_pct_short < 20 THEN 'STABLE'
        WHEN overall_avg_duration > 10 AND avg_pct_short < 40 THEN 'MODERATE'
        ELSE 'UNSTABLE'
    END as stability_rating
FROM stability_metrics
ORDER BY overall_avg_duration DESC;
```

**Quality Targets:**
- Average regime duration >10-15 minutes for stable operation
- <30% of regimes lasting <5 minutes (reduces noise)
- Multiple regimes actively occurring (not dominated by one state)

### Step 1.3: Predictive Power & Market Differentiation Test

**Motivation:** The most important test - do the classifier's regime states actually correspond to meaningfully different market conditions? A good classifier should show clear differences in volatility, directional bias, or other market characteristics across its states.

```sql
-- Test if regimes predict different market conditions
WITH market_metrics AS (
    SELECT 
        c.classifier_id,
        c.regime,
        c.confidence,
        c.timestamp,
        m.close_price,
        -- Calculate forward-looking returns for predictive testing
        LEAD(m.close_price, 5) OVER (ORDER BY m.timestamp) / m.close_price - 1 as next_5min_return,
        LEAD(m.close_price, 15) OVER (ORDER BY m.timestamp) / m.close_price - 1 as next_15min_return,
        -- Volatility measures
        (m.high_price - m.low_price) / m.close_price as price_range_pct,
        m.volume / NULLIF(AVG(m.volume) OVER (ORDER BY m.timestamp ROWS 20 PRECEDING), 0) as volume_ratio
    FROM classifier_states c
    JOIN market_data m ON c.timestamp = m.timestamp
    WHERE c.confidence > 0.7  -- High confidence classifications only
)
SELECT 
    classifier_id,
    regime,
    COUNT(*) as sample_size,
    -- Return characteristics by regime
    AVG(next_5min_return) * 100 as avg_5min_return_bps,
    STDDEV(next_5min_return) * 100 as vol_5min_return_bps,
    AVG(next_15min_return) * 100 as avg_15min_return_bps,
    STDDEV(next_15min_return) * 100 as vol_15min_return_bps,
    -- Market condition differences
    AVG(price_range_pct) * 100 as avg_intrabar_vol_bps,
    AVG(volume_ratio) as avg_volume_ratio,
    -- Directional bias detection
    COUNT(CASE WHEN next_5min_return > 0 THEN 1 END) * 100.0 / COUNT(*) as pct_positive_5min,
    -- Volatility clustering
    STDDEV(STDDEV(next_5min_return)) OVER (PARTITION BY classifier_id) as regime_vol_differentiation
FROM market_metrics
GROUP BY classifier_id, regime
HAVING COUNT(*) > 50  -- Sufficient sample size for statistical significance
ORDER BY classifier_id, avg_5min_return_bps DESC;

-- Regime differentiation score
WITH regime_stats AS (
    SELECT 
        classifier_id,
        -- Measure how different regimes are from each other
        STDDEV(avg_5min_return_bps) as return_differentiation,
        STDDEV(vol_5min_return_bps) as vol_differentiation,
        STDDEV(avg_volume_ratio) as volume_differentiation,
        COUNT(DISTINCT regime) as num_regimes
    FROM predictive_power_analysis
    GROUP BY classifier_id
)
SELECT 
    classifier_id,
    return_differentiation,
    vol_differentiation, 
    volume_differentiation,
    num_regimes,
    -- Combined differentiation score
    (return_differentiation + vol_differentiation + volume_differentiation) / 3 as overall_differentiation,
    CASE 
        WHEN (return_differentiation + vol_differentiation + volume_differentiation) / 3 > 50 THEN 'HIGH_SIGNAL'
        WHEN (return_differentiation + vol_differentiation + volume_differentiation) / 3 > 20 THEN 'MODERATE_SIGNAL'
        ELSE 'LOW_SIGNAL'
    END as signal_quality
FROM regime_stats
ORDER BY overall_differentiation DESC;
```

**Validation Criteria:**
- Clear differences in forward returns across regimes (>20 bps differentiation)
- Distinct volatility patterns between regimes  
- Volume behavior differences indicating genuine market structure changes
- Sufficient sample sizes (>50 observations per regime) for statistical confidence

### Step 1.4: Regime Transition Analysis

**Motivation:** Understanding how classifiers behave during regime transitions is crucial for real-time trading. Some strategies may need special handling during transition periods.

```sql
-- Analyze performance during regime transitions
WITH regime_transitions AS (
    SELECT 
        classifier_id,
        timestamp,
        regime as current_regime,
        LAG(regime) OVER (PARTITION BY classifier_id ORDER BY timestamp) as prev_regime,
        LEAD(regime) OVER (PARTITION BY classifier_id ORDER BY timestamp) as next_regime
    FROM classifier_states
    WHERE confidence > 0.7
),
transition_windows AS (
    SELECT 
        classifier_id,
        timestamp,
        current_regime,
        prev_regime,
        CASE 
            WHEN current_regime != prev_regime THEN 'TRANSITION_IN'
            WHEN current_regime != next_regime THEN 'TRANSITION_OUT'
            ELSE 'STABLE'
        END as transition_state
    FROM regime_transitions
),
transition_analysis AS (
    SELECT 
        tw.classifier_id,
        tw.transition_state,
        COUNT(*) as period_count,
        AVG(m.volume_ratio) as avg_volume_ratio,
        STDDEV(m.return_1min) * 100 as volatility_bps,
        -- Transition frequency
        COUNT(CASE WHEN transition_state != 'STABLE' THEN 1 END) * 100.0 / COUNT(*) as transition_frequency_pct
    FROM transition_windows tw
    JOIN market_data m ON tw.timestamp = m.timestamp
    GROUP BY tw.classifier_id, tw.transition_state
)
SELECT 
    classifier_id,
    transition_state,
    period_count,
    avg_volume_ratio,
    volatility_bps,
    transition_frequency_pct,
    -- Quality assessment
    CASE 
        WHEN transition_frequency_pct < 10 AND transition_state != 'STABLE' THEN 'CLEAN_TRANSITIONS'
        WHEN transition_frequency_pct < 20 THEN 'MODERATE_TRANSITIONS'
        ELSE 'NOISY_TRANSITIONS'
    END as transition_quality
FROM transition_analysis
ORDER BY classifier_id, transition_state;
```

## Phase 2: Classifier Ranking & Selection

After validating individual classifier quality, we need to rank and select the best performers for further analysis.

### Step 2.1: Comprehensive Classifier Scorecard

**Motivation:** Combine all quality metrics into a single scoring system to rank classifiers objectively. This helps filter from 100+ combinations down to the top 10-20 for detailed strategy analysis.

```sql
-- Create comprehensive classifier quality scores
WITH classifier_metrics AS (
    SELECT 
        classifier_id,
        -- Distribution quality (0-1 score, 1 = perfectly balanced)
        1.0 - (STDDEV(state_pct) / 25.0) as distribution_score,
        -- Stability quality (0-1 score, 1 = very stable regimes)
        LEAST(AVG(avg_duration_min) / 30.0, 1.0) as stability_score,
        -- Predictive power (0-1 score, 1 = high differentiation)
        LEAST(overall_differentiation / 100.0, 1.0) as prediction_score,
        -- Sample adequacy (0-1 score, 1 = lots of data)
        LEAST(total_observations / 10000.0, 1.0) as sample_score
    FROM (
        -- Combine results from previous analyses
        SELECT 
            d.classifier_id,
            d.pct_std as state_pct_std,
            s.overall_avg_duration as avg_duration_min,
            p.overall_differentiation,
            p.total_observations
        FROM distribution_stats d
        JOIN stability_metrics s ON d.classifier_id = s.classifier_id
        JOIN regime_stats p ON d.classifier_id = p.classifier_id
    ) combined_stats
    GROUP BY classifier_id
)
SELECT 
    classifier_id,
    distribution_score,
    stability_score, 
    prediction_score,
    sample_score,
    -- Weighted composite score (adjust weights based on importance)
    (distribution_score * 0.25 + 
     stability_score * 0.30 + 
     prediction_score * 0.30 + 
     sample_score * 0.15) as total_quality_score,
    -- Grade classification
    CASE 
        WHEN (distribution_score * 0.25 + stability_score * 0.30 + prediction_score * 0.30 + sample_score * 0.15) > 0.8 THEN 'A'
        WHEN (distribution_score * 0.25 + stability_score * 0.30 + prediction_score * 0.30 + sample_score * 0.15) > 0.6 THEN 'B'
        WHEN (distribution_score * 0.25 + stability_score * 0.30 + prediction_score * 0.30 + sample_score * 0.15) > 0.4 THEN 'C'
        ELSE 'D'
    END as classifier_grade
FROM classifier_metrics
WHERE distribution_score > 0.5  -- Minimum distribution quality threshold
  AND stability_score > 0.3     -- Minimum stability threshold
ORDER BY total_quality_score DESC;

-- Select top classifiers for strategy analysis
CREATE TABLE top_classifiers AS
SELECT classifier_id, total_quality_score, classifier_grade
FROM classifier_scorecard
WHERE classifier_grade IN ('A', 'B')  -- Keep only high-quality classifiers
  AND total_quality_score > 0.6
ORDER BY total_quality_score DESC
LIMIT 15;  -- Focus on top 15 classifiers for strategy analysis
```

**Selection Criteria:**
- Composite score >0.6 (good overall quality)
- Grade A or B classification
- Minimum thresholds on individual components (no fatal flaws)

### Step 2.2: Regime Stability Monitoring

**Motivation:** Monitor regime stability to determine when ensemble rebalancing may be needed.

```sql
-- Monitor regime stability for rebalancing decisions
WITH regime_stability_rolling AS (
    SELECT 
        classifier_id,
        timestamp,
        regime,
        -- Calculate rolling regime persistence
        COUNT(*) OVER (
            PARTITION BY classifier_id, regime 
            ORDER BY timestamp 
            ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
        ) as regime_persistence_30min
    FROM classifier_states
)
SELECT 
    classifier_id,
    AVG(regime_persistence_30min) as avg_persistence,
    STDDEV(regime_persistence_30min) as persistence_volatility,
    -- Rebalancing urgency score
    CASE 
        WHEN STDDEV(regime_persistence_30min) > 10 THEN 'HIGH_URGENCY'
        WHEN STDDEV(regime_persistence_30min) > 5 THEN 'MODERATE_URGENCY'
        ELSE 'LOW_URGENCY'
    END as rebalancing_urgency
FROM regime_stability_rolling
GROUP BY classifier_id;
```

## Phase 3: Strategy Performance Analysis by Regime

Now we analyze how the 1000+ strategy combinations perform under different regime conditions identified by our top classifiers.

### Step 3.1: Strategy-Regime Performance Matrix

**Motivation:** Different strategies should excel in different market regimes. A momentum strategy might work well in trending regimes but poorly in ranging markets. We need to quantify these relationships to build optimal ensembles.

```sql
-- Comprehensive strategy performance by regime
WITH strategy_signals_enriched AS (
    SELECT 
        s.strategy_id,
        s.timestamp,
        s.signal_type,
        s.entry_price,
        s.exit_price,
        s.return_pct,
        s.duration_minutes,
        c.classifier_id,
        c.regime,
        c.confidence,
        -- Contextual information
        EXTRACT(hour FROM s.timestamp) as trade_hour,
        EXTRACT(dow FROM s.timestamp) as day_of_week,
        ROW_NUMBER() OVER (PARTITION BY s.strategy_id ORDER BY s.timestamp) as trade_sequence
    FROM signals s
    JOIN classifier_states c ON s.timestamp = c.timestamp
    WHERE c.classifier_id IN (SELECT classifier_id FROM top_classifiers)
      AND c.confidence > 0.7  -- High-confidence regime classifications only
      AND s.return_pct IS NOT NULL  -- Complete trades only
)
SELECT 
    strategy_id,
    classifier_id,
    regime,
    -- Basic performance metrics
    COUNT(*) as total_trades,
    AVG(return_pct) * 100 as avg_return_bps,
    STDDEV(return_pct) * 100 as return_vol_bps,
    -- Risk-adjusted returns
    CASE 
        WHEN STDDEV(return_pct) > 0 THEN AVG(return_pct) / STDDEV(return_pct) 
        ELSE 0 
    END as sharpe_ratio,
    -- Win rate metrics
    COUNT(CASE WHEN return_pct > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate_pct,
    AVG(CASE WHEN return_pct > 0 THEN return_pct ELSE 0 END) * 100 as avg_win_bps,
    AVG(CASE WHEN return_pct < 0 THEN ABS(return_pct) ELSE 0 END) * 100 as avg_loss_bps,
    -- Consistency metrics
    COUNT(CASE WHEN return_pct > 0 THEN 1 END) / NULLIF(COUNT(CASE WHEN return_pct < 0 THEN 1 END), 0) as win_loss_ratio,
    MAX(return_pct) * 100 as max_win_bps,
    MIN(return_pct) * 100 as max_loss_bps,
    -- Trading frequency
    AVG(duration_minutes) as avg_trade_duration,
    COUNT(*) / NULLIF(COUNT(DISTINCT DATE(timestamp)), 0) as trades_per_day
FROM strategy_signals_enriched
GROUP BY strategy_id, classifier_id, regime
HAVING COUNT(*) >= 20  -- Minimum trades for statistical significance
ORDER BY strategy_id, classifier_id, sharpe_ratio DESC;
```

### Step 3.2: Strategy-Regime Fit Analysis

**Motivation:** Identify strategies that have clear regime preferences - those that significantly outperform their baseline when in specific market conditions. These become the building blocks of our regime-aware ensemble.

```sql
-- Find strategies with clear regime preferences
WITH strategy_baseline AS (
    -- Calculate each strategy's overall performance as baseline
    SELECT 
        strategy_id,
        AVG(return_pct) * 100 as baseline_return_bps,
        STDDEV(return_pct) * 100 as baseline_vol_bps,
        AVG(return_pct) / NULLIF(STDDEV(return_pct), 0) as baseline_sharpe
    FROM signals
    WHERE return_pct IS NOT NULL
    GROUP BY strategy_id
),
regime_performance AS (
    SELECT 
        spm.*,
        sb.baseline_return_bps,
        sb.baseline_sharpe,
        -- Calculate regime-specific alpha
        spm.avg_return_bps - sb.baseline_return_bps as regime_alpha_bps,
        spm.sharpe_ratio - sb.baseline_sharpe as regime_sharpe_alpha,
        -- Rank within regime
        RANK() OVER (PARTITION BY spm.classifier_id, spm.regime ORDER BY spm.sharpe_ratio DESC) as regime_rank,
        -- Percentile performance within strategy
        PERCENT_RANK() OVER (PARTITION BY spm.strategy_id ORDER BY spm.sharpe_ratio) as performance_percentile
    FROM strategy_performance_matrix spm
    JOIN strategy_baseline sb ON spm.strategy_id = sb.strategy_id
)
SELECT 
    strategy_id,
    classifier_id,
    regime,
    total_trades,
    avg_return_bps,
    sharpe_ratio,
    baseline_sharpe,
    regime_alpha_bps,
    regime_sharpe_alpha,
    regime_rank,
    performance_percentile,
    -- Classification of regime fit
    CASE 
        WHEN regime_sharpe_alpha > 0.5 AND regime_rank <= 5 THEN 'STRONG_FIT'
        WHEN regime_sharpe_alpha > 0.2 AND regime_rank <= 10 THEN 'GOOD_FIT'
        WHEN regime_sharpe_alpha > 0 THEN 'WEAK_FIT'
        ELSE 'POOR_FIT'
    END as regime_fit_quality,
    -- Statistical significance (simplified)
    CASE 
        WHEN total_trades >= 50 AND regime_sharpe_alpha > 0.3 THEN 'SIGNIFICANT'
        WHEN total_trades >= 30 AND regime_sharpe_alpha > 0.5 THEN 'LIKELY_SIGNIFICANT'
        ELSE 'INSUFFICIENT_DATA'
    END as significance_level
FROM regime_performance
WHERE regime_sharpe_alpha > 0  -- Only positive alpha strategies
ORDER BY classifier_id, regime, regime_sharpe_alpha DESC;

-- Summary: Best strategy candidates for each regime
SELECT 
    classifier_id,
    regime,
    COUNT(*) as candidate_strategies,
    AVG(regime_sharpe_alpha) as avg_alpha,
    STRING_AGG(strategy_id::text, ', ' ORDER BY regime_sharpe_alpha DESC LIMIT 5) as top_5_strategies
FROM regime_performance
WHERE regime_fit_quality IN ('STRONG_FIT', 'GOOD_FIT')
  AND significance_level IN ('SIGNIFICANT', 'LIKELY_SIGNIFICANT')
GROUP BY classifier_id, regime
ORDER BY classifier_id, regime;
```

**Key Insights to Extract:**
- Which strategies consistently outperform in specific regimes
- How much alpha can be captured through regime-aware selection
- Which regimes have the most strategy options vs. limited choices
- Statistical confidence in regime-strategy relationships

### Step 3.3: Time-of-Day Performance Analysis

**Motivation:** For 1-minute trading, session effects are crucial. Analyze how strategies perform during different market sessions and within different regimes.

```sql
-- Analyze performance by time of day
WITH time_segmented_performance AS (
    SELECT 
        s.strategy_id,
        EXTRACT(hour FROM s.timestamp) as hour,
        CASE 
            WHEN EXTRACT(hour FROM s.timestamp) BETWEEN 9 AND 10 THEN 'OPEN'
            WHEN EXTRACT(hour FROM s.timestamp) BETWEEN 10 AND 12 THEN 'MORNING'
            WHEN EXTRACT(hour FROM s.timestamp) BETWEEN 12 AND 14 THEN 'MIDDAY'
            WHEN EXTRACT(hour FROM s.timestamp) BETWEEN 14 AND 16 THEN 'CLOSE'
        END as session_period,
        s.return_pct,
        c.regime,
        c.classifier_id
    FROM signals s
    JOIN classifier_states c ON s.timestamp = c.timestamp
    WHERE c.classifier_id IN (SELECT classifier_id FROM top_classifiers)
)
SELECT 
    strategy_id,
    classifier_id,
    session_period,
    regime,
    COUNT(*) as trades,
    AVG(return_pct) * 100 as avg_return_bps,
    STDDEV(return_pct) * 100 as vol_bps,
    AVG(return_pct) / NULLIF(STDDEV(return_pct), 0) as sharpe_ratio,
    -- Session-specific metrics
    COUNT(CASE WHEN return_pct > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate,
    MAX(return_pct) * 100 as best_trade_bps,
    MIN(return_pct) * 100 as worst_trade_bps
FROM time_segmented_performance
GROUP BY strategy_id, classifier_id, session_period, regime
HAVING COUNT(*) > 10
ORDER BY strategy_id, classifier_id, session_period, sharpe_ratio DESC;

-- Session performance summary
SELECT 
    session_period,
    regime,
    COUNT(DISTINCT strategy_id) as active_strategies,
    AVG(sharpe_ratio) as avg_sharpe,
    COUNT(CASE WHEN sharpe_ratio > 0.5 THEN 1 END) as high_performing_strategies
FROM time_of_day_analysis
GROUP BY session_period, regime
ORDER BY session_period, avg_sharpe DESC;
```

## Phase 4: Strategy Quality Filtering

Before building ensembles, we need additional filters to ensure we're only working with robust, executable strategies.

### Step 4.1: Trade Frequency Analysis

**Motivation:** Strategies that trade too infrequently won't provide enough signal for reliable regime-based allocation. For 1-minute data, we need strategies that generate sufficient trades to be statistically meaningful and practically useful.

```sql
-- Analyze strategy trade frequency and patterns
WITH strategy_frequency AS (
    SELECT 
        strategy_id,
        COUNT(*) as total_trades,
        COUNT(DISTINCT DATE(timestamp)) as trading_days,
        COUNT(*) / NULLIF(COUNT(DISTINCT DATE(timestamp)), 0) as trades_per_day,
        -- Trading intensity patterns
        AVG(duration_minutes) as avg_trade_duration,
        STDDEV(duration_minutes) as duration_consistency,
        -- Time distribution analysis
        COUNT(CASE WHEN EXTRACT(hour FROM timestamp) BETWEEN 9 AND 11 THEN 1 END) as morning_trades,
        COUNT(CASE WHEN EXTRACT(hour FROM timestamp) BETWEEN 11 AND 14 THEN 1 END) as midday_trades,
        COUNT(CASE WHEN EXTRACT(hour FROM timestamp) BETWEEN 14 AND 16 THEN 1 END) as afternoon_trades,
        -- Gap analysis (time between trades)
        AVG(timestamp - LAG(timestamp) OVER (PARTITION BY strategy_id ORDER BY timestamp)) as avg_trade_gap,
        -- Recent activity (last 30 days)
        COUNT(CASE WHEN timestamp >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_trades
    FROM signals
    WHERE return_pct IS NOT NULL
    GROUP BY strategy_id
),
frequency_classification AS (
    SELECT 
        *,
        CASE 
            WHEN trades_per_day >= 10 THEN 'HIGH_FREQUENCY'
            WHEN trades_per_day >= 3 THEN 'MEDIUM_FREQUENCY'  
            WHEN trades_per_day >= 1 THEN 'LOW_FREQUENCY'
            ELSE 'VERY_LOW_FREQUENCY'
        END as frequency_class,
        CASE 
            WHEN total_trades >= 500 AND trades_per_day >= 2 THEN 'SUFFICIENT'
            WHEN total_trades >= 200 AND trades_per_day >= 1 THEN 'MARGINAL'
            ELSE 'INSUFFICIENT'
        END as sample_adequacy
    FROM strategy_frequency
)
SELECT 
    strategy_id,
    total_trades,
    trades_per_day,
    frequency_class,
    sample_adequacy,
    avg_trade_duration,
    avg_trade_gap,
    recent_trades,
    -- Activity distribution score (prefer strategies active throughout day)
    (morning_trades + midday_trades + afternoon_trades) / 3.0 / GREATEST(morning_trades, midday_trades, afternoon_trades) as activity_balance
FROM frequency_classification
WHERE sample_adequacy IN ('SUFFICIENT', 'MARGINAL')  -- Filter out low-frequency strategies
ORDER BY trades_per_day DESC;
```

### Step 4.2: Parameter Sensitivity Analysis

**Motivation:** Robust strategies should work across a neighborhood of similar parameters, not just at one specific point. This helps identify genuine edges vs. overfitted parameter combinations.

```sql
-- Parameter sensitivity and return clustering analysis
WITH strategy_parameters AS (
    -- Extract parameter values from strategy metadata
    SELECT 
        strategy_id,
        strategy_type,
        JSON_EXTRACT(parameters, '$.lookback_period') as lookback_period,
        JSON_EXTRACT(parameters, '$.threshold') as threshold,
        JSON_EXTRACT(parameters, '$.ma_fast') as ma_fast,
        JSON_EXTRACT(parameters, '$.ma_slow') as ma_slow,
        -- Add other relevant parameters based on your strategy types
        AVG(return_pct) * 100 as avg_return_bps,
        STDDEV(return_pct) * 100 as return_vol_bps,
        AVG(return_pct) / NULLIF(STDDEV(return_pct), 0) as sharpe_ratio,
        COUNT(*) as trade_count
    FROM signals s
    GROUP BY strategy_id, strategy_type, parameters
    HAVING COUNT(*) >= 50  -- Minimum trades for analysis
),
parameter_neighborhoods AS (
    -- Find strategies with similar parameters
    SELECT 
        s1.strategy_id as base_strategy,
        s2.strategy_id as neighbor_strategy,
        s1.sharpe_ratio as base_sharpe,
        s2.sharpe_ratio as neighbor_sharpe,
        -- Parameter distance calculation (adjust based on your parameter types)
        ABS(s1.lookback_period - s2.lookback_period) + 
        ABS(s1.threshold - s2.threshold) * 100 +  -- Scale threshold differences
        ABS(s1.ma_fast - s2.ma_fast) + 
        ABS(s1.ma_slow - s2.ma_slow) as parameter_distance
    FROM strategy_parameters s1
    JOIN strategy_parameters s2 ON s1.strategy_type = s2.strategy_type 
        AND s1.strategy_id != s2.strategy_id
    WHERE ABS(s1.lookback_period - s2.lookback_period) <= 5  -- Nearby parameters only
      AND ABS(s1.threshold - s2.threshold) <= 0.01
      AND ABS(s1.ma_fast - s2.ma_fast) <= 3
      AND ABS(s1.ma_slow - s2.ma_slow) <= 5
),
sensitivity_analysis AS (
    SELECT 
        base_strategy,
        COUNT(*) as neighbor_count,
        AVG(neighbor_sharpe) as avg_neighbor_sharpe,
        STDDEV(neighbor_sharpe) as neighbor_sharpe_std,
        base_sharpe,
        -- Sensitivity metrics
        base_sharpe - AVG(neighbor_sharpe) as sharpe_advantage,
        CASE 
            WHEN STDDEV(neighbor_sharpe) > 0 
            THEN (base_sharpe - AVG(neighbor_sharpe)) / STDDEV(neighbor_sharpe)
            ELSE 0 
        END as sharpe_z_score,
        -- Robustness score
        COUNT(CASE WHEN neighbor_sharpe > 0 THEN 1 END) * 1.0 / COUNT(*) as positive_neighbor_pct
    FROM parameter_neighborhoods
    GROUP BY base_strategy, base_sharpe
    HAVING COUNT(*) >= 5  -- Need sufficient neighbors for analysis
)
SELECT 
    base_strategy,
    base_sharpe,
    neighbor_count,
    avg_neighbor_sharpe,
    neighbor_sharpe_std,
    sharpe_advantage,
    sharpe_z_score,
    positive_neighbor_pct,
    CASE 
        WHEN positive_neighbor_pct > 0.7 AND sharpe_z_score > 1.0 THEN 'ROBUST'
        WHEN positive_neighbor_pct > 0.5 AND sharpe_z_score > 0.5 THEN 'MODERATE'
        ELSE 'FRAGILE'
    END as robustness_classification
FROM sensitivity_analysis
WHERE base_sharpe > 0  -- Only consider profitable base strategies
ORDER BY sharpe_z_score DESC;
```

### Step 4.3: Execution Cost Robustness Analysis

**Motivation:** Strategies must remain profitable after realistic transaction costs, slippage, and market impact. This is especially critical for 1-minute trading where costs can quickly erode edge.

```sql
-- Transaction cost impact analysis
WITH cost_scenarios AS (
    SELECT 
        strategy_id,
        return_pct,
        duration_minutes,
        -- Different cost scenarios (adjust based on your broker/market)
        CASE 
            WHEN duration_minutes <= 5 THEN 0.0003  -- 3 bps for very short trades (higher impact)
            WHEN duration_minutes <= 15 THEN 0.0002  -- 2 bps for short trades
            WHEN duration_minutes <= 60 THEN 0.0001  -- 1 bp for medium trades
            ELSE 0.00005  -- 0.5 bps for longer trades
        END as estimated_cost_pct,
        -- Volume-based cost estimation (if volume data available)
        CASE 
            WHEN volume_ratio > 2.0 THEN 0.0001  -- Extra cost for high volume periods
            ELSE 0
        END as liquidity_cost_pct
    FROM signals s
    LEFT JOIN (
        SELECT timestamp, volume / AVG(volume) OVER (ORDER BY timestamp ROWS 100 PRECEDING) as volume_ratio
        FROM market_data
    ) v ON s.timestamp = v.timestamp
    WHERE return_pct IS NOT NULL
),
cost_adjusted_performance AS (
    SELECT 
        strategy_id,
        COUNT(*) as total_trades,
        -- Gross performance
        AVG(return_pct) * 100 as gross_return_bps,
        STDDEV(return_pct) * 100 as gross_vol_bps,
        AVG(return_pct) / NULLIF(STDDEV(return_pct), 0) as gross_sharpe,
        -- Cost estimates
        AVG(estimated_cost_pct + liquidity_cost_pct) * 100 as avg_cost_bps,
        -- Net performance
        AVG(return_pct - estimated_cost_pct - liquidity_cost_pct) * 100 as net_return_bps,
        STDDEV(return_pct - estimated_cost_pct - liquidity_cost_pct) * 100 as net_vol_bps,
        AVG(return_pct - estimated_cost_pct - liquidity_cost_pct) / 
            NULLIF(STDDEV(return_pct - estimated_cost_pct - liquidity_cost_pct), 0) as net_sharpe,
        -- Cost impact metrics
        (AVG(return_pct) - AVG(return_pct - estimated_cost_pct - liquidity_cost_pct)) * 100 as cost_drag_bps,
        COUNT(CASE WHEN return_pct - estimated_cost_pct - liquidity_cost_pct > 0 THEN 1 END) * 1.0 / COUNT(*) as net_win_rate
    FROM cost_scenarios
    GROUP BY strategy_id
)
SELECT 
    strategy_id,
    total_trades,
    gross_return_bps,
    net_return_bps,
    cost_drag_bps,
    gross_sharpe,
    net_sharpe,
    gross_sharpe - net_sharpe as sharpe_degradation,
    net_win_rate,
    CASE 
        WHEN net_sharpe > 0.5 AND net_return_bps > 10 THEN 'COST_ROBUST'
        WHEN net_sharpe > 0.2 AND net_return_bps > 5 THEN 'COST_SENSITIVE'
        ELSE 'COST_PROHIBITIVE'
    END as cost_robustness
FROM cost_adjusted_performance
WHERE gross_sharpe > 0  -- Only analyze profitable strategies
ORDER BY net_sharpe DESC;
```

### Step 4.4: Strategy Correlation Analysis & Diversification

**Motivation:** Select strategies that provide complementary signals rather than redundant ones. High correlation between strategies reduces ensemble effectiveness and increases concentration risk.

```sql
-- Strategy correlation and diversification analysis
WITH strategy_returns AS (
    -- Align all strategy returns to common time grid
    SELECT 
        timestamp,
        strategy_id,
        return_pct
    FROM signals
    WHERE return_pct IS NOT NULL
      AND strategy_id IN (SELECT strategy_id FROM qualified_strategies)  -- Pre-filtered strategies
),
return_matrix AS (
    -- Pivot to get side-by-side returns for correlation calculation
    SELECT 
        timestamp,
        SUM(CASE WHEN strategy_id = 'strategy_1' THEN return_pct END) as ret_1,
        SUM(CASE WHEN strategy_id = 'strategy_2' THEN return_pct END) as ret_2,
        -- Add more strategy columns as needed, or use dynamic SQL
        COUNT(DISTINCT strategy_id) as active_strategies
    FROM strategy_returns
    GROUP BY timestamp
    HAVING COUNT(DISTINCT strategy_id) >= 2  -- Need multiple strategies active
),
correlation_pairs AS (
    SELECT 
        s1.strategy_id as strategy_a,
        s2.strategy_id as strategy_b,
        CORR(s1.return_pct, s2.return_pct) as correlation,
        COUNT(*) as overlapping_trades,
        -- Return characteristics
        AVG(s1.return_pct) as avg_return_a,
        AVG(s2.return_pct) as avg_return_b,
        STDDEV(s1.return_pct) as vol_a,
        STDDEV(s2.return_pct) as vol_b
    FROM strategy_returns s1
    JOIN strategy_returns s2 ON s1.timestamp = s2.timestamp 
        AND s1.strategy_id < s2.strategy_id  -- Avoid duplicate pairs
    GROUP BY s1.strategy_id, s2.strategy_id
    HAVING COUNT(*) >= 20  -- Minimum overlapping observations
),
diversification_analysis AS (
    SELECT 
        strategy_a,
        AVG(correlation) as avg_correlation_with_others,
        MAX(correlation) as max_correlation,
        COUNT(*) as correlation_pairs,
        COUNT(CASE WHEN correlation > 0.7 THEN 1 END) as high_corr_count,
        COUNT(CASE WHEN correlation < 0.3 THEN 1 END) as low_corr_count,
        -- Diversification score
        1.0 - AVG(ABS(correlation)) as diversification_score
    FROM correlation_pairs
    GROUP BY strategy_a
),
strategy_clusters AS (
    -- Identify clusters of highly correlated strategies
    SELECT 
        strategy_a,
        strategy_b,
        correlation,
        CASE 
            WHEN correlation > 0.8 THEN 'HIGHLY_CORRELATED'
            WHEN correlation > 0.5 THEN 'MODERATELY_CORRELATED'
            WHEN correlation < -0.5 THEN 'NEGATIVELY_CORRELATED'
            ELSE 'UNCORRELATED'
        END as correlation_category
    FROM correlation_pairs
)
SELECT 
    da.strategy_a as strategy_id,
    da.avg_correlation_with_others,
    da.max_correlation,
    da.diversification_score,
    da.high_corr_count,
    da.low_corr_count,
    CASE 
        WHEN da.diversification_score > 0.7 AND da.max_correlation < 0.6 THEN 'UNIQUE'
        WHEN da.diversification_score > 0.5 AND da.high_corr_count <= 2 THEN 'DIVERSIFYING'
        WHEN da.high_corr_count > 5 THEN 'REDUNDANT'
        ELSE 'MODERATE'
    END as diversification_classification,
    -- Correlation summary
    STRING_AGG(
        CASE WHEN sc.correlation > 0.7 
        THEN sc.strategy_b || ' (' || ROUND(sc.correlation, 2) || ')'
        END, ', '
    ) as highly_correlated_strategies
FROM diversification_analysis da
LEFT JOIN strategy_clusters sc ON da.strategy_a = sc.strategy_a 
    AND sc.correlation_category = 'HIGHLY_CORRELATED'
GROUP BY da.strategy_a, da.avg_correlation_with_others, da.max_correlation, 
         da.diversification_score, da.high_corr_count, da.low_corr_count
ORDER BY da.diversification_score DESC;
```

### Step 4.5: Drawdown Analysis

**Motivation:** Understand the risk characteristics of strategies by analyzing their drawdown patterns across different regimes.

```sql
-- Calculate rolling drawdowns by strategy and regime
WITH cumulative_returns AS (
    SELECT 
        s.strategy_id,
        c.classifier_id,
        c.regime,
        s.timestamp,
        s.return_pct,
        SUM(s.return_pct) OVER (
            PARTITION BY s.strategy_id, c.classifier_id, c.regime 
            ORDER BY s.timestamp
        ) as cum_return,
        MAX(SUM(s.return_pct)) OVER (
            PARTITION BY s.strategy_id, c.classifier_id, c.regime 
            ORDER BY s.timestamp
            ROWS UNBOUNDED PRECEDING
        ) as running_max_return
    FROM signals s
    JOIN classifier_states c ON s.timestamp = c.timestamp
    WHERE c.classifier_id IN (SELECT classifier_id FROM top_classifiers)
      AND s.return_pct IS NOT NULL
),
drawdown_metrics AS (
    SELECT 
        strategy_id,
        classifier_id,
        regime,
        timestamp,
        cum_return,
        running_max_return,
        cum_return - running_max_return as drawdown,
        CASE 
            WHEN cum_return < running_max_return THEN 1 
            ELSE 0 
        END as in_drawdown
    FROM cumulative_returns
)
SELECT 
    strategy_id,
    classifier_id,
    regime,
    COUNT(*) as total_periods,
    MIN(drawdown) * 100 as max_drawdown_bps,
    AVG(CASE WHEN in_drawdown = 1 THEN drawdown ELSE NULL END) * 100 as avg_drawdown_bps,
    COUNT(CASE WHEN in_drawdown = 1 THEN 1 END) * 100.0 / COUNT(*) as pct_time_in_drawdown,
    -- Recovery analysis
    AVG(CASE 
        WHEN in_drawdown = 1 AND LEAD(in_drawdown) OVER (PARTITION BY strategy_id, classifier_id, regime ORDER BY timestamp) = 0 
        THEN 1 ELSE 0 
    END) as drawdown_frequency,
    -- Risk-adjusted performance
    AVG(cum_return) * 100 / NULLIF(ABS(MIN(drawdown)) * 100, 0) as return_to_max_dd_ratio
FROM drawdown_metrics
GROUP BY strategy_id, classifier_id, regime
HAVING COUNT(*) >= 50  -- Sufficient data for drawdown analysis
ORDER BY strategy_id, classifier_id, max_drawdown_bps DESC;

-- Drawdown summary by strategy
SELECT 
    strategy_id,
    AVG(max_drawdown_bps) as avg_max_dd_across_regimes,
    MAX(max_drawdown_bps) as worst_regime_dd,
    AVG(pct_time_in_drawdown) as avg_pct_time_in_dd,
    AVG(return_to_max_dd_ratio) as avg_return_dd_ratio,
    CASE 
        WHEN AVG(max_drawdown_bps) < -20 THEN 'HIGH_RISK'
        WHEN AVG(max_drawdown_bps) < -10 THEN 'MODERATE_RISK'
        ELSE 'LOW_RISK'
    END as risk_classification
FROM drawdown_analysis
GROUP BY strategy_id
ORDER BY avg_max_dd_across_regimes ASC;
```

## Phase 5: Ensemble Construction & Validation

### Step 5.1: Integrated Strategy Selection Pipeline

**Motivation:** Combine all quality filters into a comprehensive scoring system that balances performance, robustness, executability, and diversification.

```sql
-- Master strategy qualification pipeline
WITH master_strategy_scores AS (
    SELECT 
        s.strategy_id,
        -- Performance metrics
        s.gross_sharpe,
        s.net_sharpe,
        s.cost_robustness,
        -- Frequency and sample quality
        f.trades_per_day,
        f.sample_adequacy,
        f.activity_balance,
        -- Parameter robustness
        p.robustness_classification,
        p.sharpe_z_score,
        p.positive_neighbor_pct,
        -- Diversification value
        d.diversification_score,
        d.diversification_classification,
        d.max_correlation,
        -- Risk metrics
        dd.avg_max_dd_across_regimes,
        dd.risk_classification,
        -- Composite scoring
        CASE 
            WHEN s.cost_robustness = 'COST_ROBUST' THEN 1.0
            WHEN s.cost_robustness = 'COST_SENSITIVE' THEN 0.6
            ELSE 0.2
        END as cost_score,
        CASE 
            WHEN f.sample_adequacy = 'SUFFICIENT' THEN 1.0
            WHEN f.sample_adequacy = 'MARGINAL' THEN 0.6
            ELSE 0.2
        END as frequency_score,
        CASE 
            WHEN p.robustness_classification = 'ROBUST' THEN 1.0
            WHEN p.robustness_classification = 'MODERATE' THEN 0.6
            ELSE 0.2
        END as robustness_score,
        CASE 
            WHEN d.diversification_classification = 'UNIQUE' THEN 1.0
            WHEN d.diversification_classification = 'DIVERSIFYING' THEN 0.8
            WHEN d.diversification_classification = 'MODERATE' THEN 0.5
            ELSE 0.2
        END as diversification_score_normalized,
        CASE 
            WHEN dd.risk_classification = 'LOW_RISK' THEN 1.0
            WHEN dd.risk_classification = 'MODERATE_RISK' THEN 0.7
            ELSE 0.3
        END as risk_score
    FROM cost_adjusted_performance s
    JOIN frequency_classification f ON s.strategy_id = f.strategy_id
    LEFT JOIN sensitivity_analysis p ON s.strategy_id = p.base_strategy
    LEFT JOIN diversification_analysis d ON s.strategy_id = d.strategy_id
    LEFT JOIN drawdown_summary dd ON s.strategy_id = dd.strategy_id
    WHERE s.net_sharpe > 0.2  -- Minimum performance threshold
      AND f.sample_adequacy IN ('SUFFICIENT', 'MARGINAL')
),
final_strategy_ranking AS (
    SELECT 
        strategy_id,
        net_sharpe,
        cost_score,
        frequency_score,
        robustness_score,
        diversification_score_normalized,
        risk_score,
        -- Weighted composite score
        (net_sharpe * 0.30 +  -- Performance weight
         cost_score * 0.20 +  -- Cost robustness weight  
         frequency_score * 0.15 + -- Frequency weight
         robustness_score * 0.15 + -- Parameter robustness weight
         diversification_score_normalized * 0.10 + -- Diversification weight
         risk_score * 0.10) as composite_score, -- Risk weight
        -- Individual quality gates
        CASE 
            WHEN cost_score >= 0.6 AND frequency_score >= 0.6 AND 
                 robustness_score >= 0.6 AND net_sharpe > 0.3 THEN 'TIER_1'
            WHEN cost_score >= 0.6 AND frequency_score >= 0.6 AND net_sharpe > 0.2 THEN 'TIER_2'
            WHEN net_sharpe > 0.2 THEN 'TIER_3'
            ELSE 'REJECTED'
        END as strategy_tier
    FROM master_strategy_scores
)
SELECT 
    strategy_id,
    strategy_tier,
    net_sharpe,
    composite_score,
    cost_score,
    frequency_score,
    robustness_score,
    diversification_score_normalized,
    risk_score,
    -- Selection recommendations
    CASE 
        WHEN strategy_tier = 'TIER_1' THEN 'INCLUDE_PRIMARY'
        WHEN strategy_tier = 'TIER_2' AND diversification_score_normalized > 0.6 THEN 'INCLUDE_SECONDARY'
        WHEN strategy_tier = 'TIER_3' AND diversification_score_normalized = 1.0 THEN 'INCLUDE_DIVERSIFIER'
        ELSE 'EXCLUDE'
    END as selection_recommendation
FROM final_strategy_ranking
WHERE strategy_tier != 'REJECTED'
ORDER BY composite_score DESC;

-- Summary statistics for qualified strategies
SELECT 
    strategy_tier,
    COUNT(*) as strategy_count,
    AVG(net_sharpe) as avg_net_sharpe,
    AVG(composite_score) as avg_composite_score,
    MIN(net_sharpe) as min_net_sharpe,
    MAX(net_sharpe) as max_net_sharpe
FROM final_strategy_ranking
WHERE strategy_tier != 'REJECTED'
GROUP BY strategy_tier
ORDER BY strategy_tier;
```

### Step 5.2: Regime-Strategy Allocation Matrix

**Motivation:** Build the final allocation matrix that determines which qualified strategies to use under each classifier regime, incorporating all quality filters.

```sql
-- Create final regime-strategy allocation matrix
WITH qualified_strategies AS (
    SELECT strategy_id 
    FROM final_strategy_ranking 
    WHERE selection_recommendation LIKE 'INCLUDE%'
),
regime_strategy_performance AS (
    -- Re-run regime performance analysis with only qualified strategies
    SELECT 
        spm.strategy_id,
        spm.classifier_id,
        spm.regime,
        spm.sharpe_ratio,
        spm.total_trades,
        spm.avg_return_bps,
        spm.net_sharpe,  -- From cost analysis
        fsr.strategy_tier,
        fsr.composite_score,
        -- Rank within regime
        RANK() OVER (PARTITION BY spm.classifier_id, spm.regime ORDER BY spm.sharpe_ratio DESC) as regime_rank,
        -- Performance relative to strategy's baseline
        spm.sharpe_ratio - AVG(spm.sharpe_ratio) OVER (PARTITION BY spm.strategy_id) as regime_alpha
    FROM strategy_performance_matrix spm
    JOIN qualified_strategies qs ON spm.strategy_id = qs.strategy_id
    JOIN final_strategy_ranking fsr ON spm.strategy_id = fsr.strategy_id
    WHERE spm.total_trades >= 20  -- Minimum sample size
),
allocation_weights AS (
    SELECT 
        classifier_id,
        regime,
        strategy_id,
        sharpe_ratio,
        regime_rank,
        strategy_tier,
        composite_score,
        -- Calculate allocation weights based on performance and quality
        CASE 
            WHEN regime_rank <= 3 AND strategy_tier = 'TIER_1' THEN sharpe_ratio * 1.5  -- Boost top performers
            WHEN regime_rank <= 5 AND strategy_tier IN ('TIER_1', 'TIER_2') THEN sharpe_ratio * 1.2
            WHEN regime_rank <= 10 THEN sharpe_ratio
            ELSE sharpe_ratio * 0.8  -- Reduce weight for lower performers
        END as weighted_score
    FROM regime_strategy_performance
    WHERE regime_alpha > 0  -- Only strategies that outperform in this regime
      AND sharpe_ratio > 0.3  -- Minimum performance threshold
),
normalized_allocations AS (
    SELECT 
        classifier_id,
        regime,
        strategy_id,
        weighted_score,
        regime_rank,
        strategy_tier,
        -- Normalize weights to sum to 1 within each regime
        weighted_score / SUM(weighted_score) OVER (PARTITION BY classifier_id, regime) as allocation_weight,
        -- Limit maximum allocation to prevent concentration
        LEAST(
            weighted_score / SUM(weighted_score) OVER (PARTITION BY classifier_id, regime),
            0.4  -- Maximum 40% allocation to any single strategy
        ) as capped_allocation_weight
    FROM allocation_weights
)
SELECT 
    classifier_id,
    regime,
    strategy_id,
    regime_rank,
    strategy_tier,
    allocation_weight,
    capped_allocation_weight,
    -- Re-normalize after capping
    capped_allocation_weight / SUM(capped_allocation_weight) OVER (PARTITION BY classifier_id, regime) as final_allocation_weight,
    -- Strategy count per regime
    COUNT(*) OVER (PARTITION BY classifier_id, regime) as strategies_in_regime
FROM normalized_allocations
WHERE capped_allocation_weight > 0.05  -- Minimum 5% allocation threshold
ORDER BY classifier_id, regime, allocation_weight DESC;

-- Allocation summary by classifier and regime
SELECT 
    classifier_id,
    regime,
    COUNT(*) as num_strategies,
    SUM(final_allocation_weight) as total_weight_check,  -- Should sum to ~1.0
    AVG(final_allocation_weight) as avg_allocation,
    MAX(final_allocation_weight) as max_allocation,
    STRING_AGG(strategy_id || ' (' || ROUND(final_allocation_weight, 2) || ')', ', ' 
               ORDER BY final_allocation_weight DESC) as top_strategies
FROM normalized_allocations
GROUP BY classifier_id, regime
ORDER BY classifier_id, regime;
```

### Step 5.3: Walk-Forward Validation Framework

**Motivation:** Prevent overfitting by testing the ensemble system on out-of-sample data using realistic trading constraints and costs.

```sql
-- Walk-forward validation setup
-- This would typically be implemented in Python, but here's the SQL framework

WITH validation_periods AS (
    -- Create rolling 6-month training, 1-month test periods
    SELECT 
        period_id,
        training_start,
        training_end,
        test_start,
        test_end
    FROM (
        SELECT 
            ROW_NUMBER() OVER (ORDER BY month_start) as period_id,
            month_start - INTERVAL '6 months' as training_start,
            month_start - INTERVAL '1 day' as training_end,
            month_start as test_start,
            month_start + INTERVAL '1 month' - INTERVAL '1 day' as test_end
        FROM (
            SELECT DISTINCT DATE_TRUNC('month', timestamp) as month_start
            FROM signals
            WHERE timestamp >= '2023-01-01'  -- Adjust date range
            ORDER BY month_start
        ) months
    ) periods
    WHERE training_start >= (SELECT MIN(timestamp) FROM signals)
),
period_results AS (
    SELECT 
        vp.period_id,
        vp.test_start,
        vp.test_end,
        -- Calculate ensemble performance in test period
        SUM(s.return_pct * ew.ensemble_weight) as ensemble_return,
        COUNT(*) as total_trades,
        STDDEV(s.return_pct * ew.ensemble_weight) as ensemble_vol
    FROM validation_periods vp
    JOIN signals s ON s.timestamp BETWEEN vp.test_start AND vp.test_end
    JOIN ensemble_weights ew ON s.strategy_id = ew.strategy_id 
        AND s.classifier_id = ew.classifier_id 
        AND s.regime = ew.regime
    WHERE ew.selection_status = 'INCLUDE'
    GROUP BY vp.period_id, vp.test_start, vp.test_end
)
SELECT 
    period_id,
    test_start,
    test_end,
    ensemble_return * 100 as return_bps,
    ensemble_vol * 100 as vol_bps,
    ensemble_return / NULLIF(ensemble_vol, 0) as sharpe_ratio,
    total_trades,
    -- Cumulative performance tracking
    SUM(ensemble_return) OVER (ORDER BY period_id) * 100 as cumulative_return_bps,
    EXP(SUM(LN(1 + ensemble_return)) OVER (ORDER BY period_id)) - 1 as compound_return
FROM period_results
ORDER BY period_id;

-- Performance summary across all validation periods
SELECT 
    COUNT(*) as validation_periods,
    AVG(return_bps) as avg_monthly_return_bps,
    STDDEV(return_bps) as monthly_return_vol_bps,
    AVG(sharpe_ratio) as avg_monthly_sharpe,
    COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate_pct,
    MAX(cumulative_return_bps) as max_cumulative_return_bps,
    MIN(cumulative_return_bps) as max_drawdown_bps
FROM period_results;
```

## Implementation Roadmap

### Immediate Next Steps (Before Paper Trading):
1. **Run Phase 1 queries** to identify problematic classifiers early
2. **Create automated scoring pipeline** to rank all 100+ classifier combinations
3. **Filter to top 15-20 classifiers** for detailed strategy analysis
4. **Execute strategy quality filters** to identify robust candidates

### Paper Trading Preparation:
1. **Build final ensemble allocation matrix** with selected strategies and classifiers
2. **Implement real-time regime detection** using top classifiers
3. **Set up position sizing logic** based on ensemble weights
4. **Create monitoring dashboard** for regime transitions and performance

### During Paper Trading:
1. **Monitor regime stability** in real-time vs. historical patterns
2. **Track strategy performance** by regime and time of day
3. **Validate transaction cost models** with actual execution data
4. **Collect slippage statistics** for model refinement

### Post Paper Trading Analysis:
1. **Compare paper vs. backtest performance** to identify execution gaps
2. **Refine cost models** based on observed slippage
3. **Adjust ensemble weights** if certain strategies underperform
4. **Validate walk-forward results** with paper trading data

## Key Performance Indicators

### Classifier KPIs:
- State distribution balance (target: std < 10%)
- Average regime duration (target: >15 minutes)
- Regime differentiation score (target: >30)
- Transition quality (target: <20% noisy transitions)

### Strategy KPIs:
- Net Sharpe ratio after costs (target: >0.5)
- Trade frequency (target: >3 trades/day)
- Parameter robustness (target: >70% profitable neighbors)
- Maximum drawdown (target: <15%)

### Ensemble KPIs:
- Overall Sharpe ratio (target: >1.0)
- Regime coverage (target: >80% of time)
- Strategy diversification (target: <0.5 avg correlation)
- Monthly win rate (target: >60%)

## Monitoring and Alerts

Set up real-time alerts for:
1. **Regime transitions** - When classifier detects regime change
2. **Unusual strategy behavior** - Performance deviation from historical
3. **Correlation spikes** - When strategies become too correlated
4. **Drawdown breaches** - When approaching risk limits
5. **Low activity periods** - When trade frequency drops significantly

This comprehensive workflow should guide you through the entire process from classifier validation to paper trading preparation and beyond.