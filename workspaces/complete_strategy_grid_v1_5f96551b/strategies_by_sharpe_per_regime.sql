-- Sort strategies by Sharpe ratio per regime
-- Analyze top performers in each market regime

PRAGMA memory_limit='4GB';
SET threads=8;

SELECT 
    '=== STRATEGIES SORTED BY SHARPE RATIO PER REGIME ===' as header;

-- Load the results from our previous analysis
-- First, let's check what data we have in the CSV
SELECT 
    '=== DATA CHECK ===' as section;

-- Let's query the results we already computed
WITH strategy_results AS (
    -- Using the data from our previous run
    -- We'll manually input the results we saw
    SELECT * FROM (VALUES
        ('macd_crossover_grid_12_26_9', 'high_vol_bearish', 1, 0.423, 0.373, NULL, 100.0, 13.0),
        ('macd_crossover_grid_12_26_9', 'neutral', 204, -0.004, -10.991, -0.039, 51.5, 14.5),
        ('macd_crossover_grid_12_26_9', 'low_vol_bullish', 1388, 0.002, -66.164, 0.025, 48.8, 12.3),
        ('macd_crossover_grid_12_26_9', 'low_vol_bearish', 1319, -0.001, -66.766, -0.006, 50.5, 12.3),
        
        ('ema_crossover_grid_7_35', 'high_vol_bearish', 1, 0.816, 0.766, NULL, 100.0, 34.0),
        ('ema_crossover_grid_7_35', 'neutral', 65, -0.006, -3.638, -0.064, 50.8, 17.3),
        ('ema_crossover_grid_7_35', 'low_vol_bearish', 580, 0.005, -25.866, 0.037, 49.5, 25.1),
        ('ema_crossover_grid_7_35', 'low_vol_bullish', 563, -0.006, -31.661, -0.040, 52.0, 23.9),
        
        ('rsi_threshold_grid_11_40', 'neutral', 156, 0.010, -6.297, 0.113, 50.6, NULL),
        ('rsi_threshold_grid_11_40', 'low_vol_bullish', 122, -0.011, -7.418, -0.073, 52.5, NULL),
        ('rsi_threshold_grid_11_40', 'low_vol_bearish', 3308, 0.001, -161.179, 0.015, 50.0, NULL)
    ) AS t(strategy_name, entry_regime, trade_count, avg_return_pct, net_return_pct, sharpe_ratio, win_rate, avg_duration_min)
)
SELECT 
    '=== HIGH VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    avg_return_pct,
    net_return_pct,
    sharpe_ratio,
    win_rate as win_rate_pct,
    avg_duration_min
FROM strategy_results
WHERE entry_regime = 'high_vol_bearish'
  AND sharpe_ratio IS NOT NULL
ORDER BY sharpe_ratio DESC;

-- Note: Both high_vol_bearish entries have NULL Sharpe (only 1 trade each)
SELECT 
    '=== NEUTRAL REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    avg_return_pct,
    net_return_pct,
    sharpe_ratio,
    win_rate as win_rate_pct,
    avg_duration_min
FROM strategy_results
WHERE entry_regime = 'neutral'
ORDER BY sharpe_ratio DESC;

SELECT 
    '=== LOW VOLATILITY BULLISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    avg_return_pct,
    net_return_pct,
    sharpe_ratio,
    win_rate as win_rate_pct,
    avg_duration_min
FROM strategy_results
WHERE entry_regime = 'low_vol_bullish'
ORDER BY sharpe_ratio DESC;

SELECT 
    '=== LOW VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    avg_return_pct,
    net_return_pct,
    sharpe_ratio,
    win_rate as win_rate_pct,
    avg_duration_min
FROM strategy_results
WHERE entry_regime = 'low_vol_bearish'
ORDER BY sharpe_ratio DESC;

-- Summary statistics
SELECT 
    '=== SHARPE RATIO SUMMARY BY REGIME ===' as summary_section;

WITH sharpe_summary AS (
    SELECT 
        entry_regime,
        COUNT(*) as strategies_count,
        COUNT(sharpe_ratio) as strategies_with_sharpe,
        MIN(sharpe_ratio) as min_sharpe,
        AVG(sharpe_ratio) as avg_sharpe,
        MAX(sharpe_ratio) as max_sharpe,
        SUM(trade_count) as total_trades
    FROM strategy_results
    GROUP BY entry_regime
)
SELECT 
    entry_regime,
    strategies_count,
    strategies_with_sharpe,
    ROUND(min_sharpe, 3) as min_sharpe,
    ROUND(avg_sharpe, 3) as avg_sharpe,
    ROUND(max_sharpe, 3) as max_sharpe,
    total_trades
FROM sharpe_summary
ORDER BY avg_sharpe DESC;

-- Now let's analyze more strategies to get a better picture
SELECT 
    '=== ANALYZING MORE STRATEGIES FOR SHARPE COMPARISON ===' as section;

-- Let's process more strategies individually
-- Ultimate Oscillator
WITH uo_analysis AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ultimate_oscillator_grid/SPY_ultimate_oscillator_grid_2_4_8_70_30.parquet')
    WHERE ts::timestamp >= '2024-03-26 00:00:00'
      AND ts::timestamp <= '2025-01-17 20:00:00'
),
uo_trades AS (
    SELECT 
        signal_time,
        signal_value,
        next_signal_time
    FROM uo_analysis
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
),
regime_data AS (
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
        LAST_VALUE(
            (SELECT val FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') r 
             WHERE r.ts::timestamp <= timestamp::timestamp + INTERVAL 4 HOUR 
             ORDER BY r.ts DESC LIMIT 1)
            IGNORE NULLS
        ) OVER (ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as current_regime
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-03-26 00:00:00'::timestamp with time zone
      AND timestamp <= '2025-01-17 20:00:00'::timestamp with time zone
),
market_data AS (
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-03-26 00:00:00'::timestamp with time zone
      AND timestamp <= '2025-01-17 20:00:00'::timestamp with time zone
),
uo_performance AS (
    SELECT 
        rd.current_regime as entry_regime,
        COUNT(*) as trade_count,
        AVG(
            CASE 
                WHEN t.signal_value = 1 THEN (m2.close - m1.close) / m1.close
                WHEN t.signal_value = -1 THEN (m1.close - m2.close) / m1.close
            END
        ) as avg_return,
        STDDEV(
            CASE 
                WHEN t.signal_value = 1 THEN (m2.close - m1.close) / m1.close
                WHEN t.signal_value = -1 THEN (m1.close - m2.close) / m1.close
            END
        ) as return_std
    FROM uo_trades t
    LEFT JOIN regime_data rd ON t.signal_time = rd.timestamp_est
    LEFT JOIN market_data m1 ON t.signal_time = m1.timestamp_est
    LEFT JOIN market_data m2 ON t.next_signal_time = m2.timestamp_est
    WHERE rd.current_regime IS NOT NULL
      AND m1.close IS NOT NULL
      AND m2.close IS NOT NULL
    GROUP BY rd.current_regime
)
SELECT 
    'ultimate_oscillator_2_4_8_70_30' as strategy_name,
    entry_regime,
    trade_count,
    ROUND(avg_return * 100, 3) as avg_return_pct,
    ROUND((avg_return * trade_count - trade_count * 0.0005) * 100, 3) as net_return_pct,
    ROUND(avg_return / NULLIF(return_std, 0), 3) as sharpe_ratio
FROM uo_performance
WHERE trade_count >= 5;