-- Sharpe Ratio Analysis by Regime
-- Using results from our previous analysis

PRAGMA memory_limit='2GB';

SELECT 
    '=== STRATEGIES SORTED BY SHARPE RATIO PER REGIME ===' as header;

-- Manually create the results table from our previous analysis
CREATE TEMP TABLE strategy_results AS
SELECT * FROM (VALUES
    ('MACD Crossover 12_26_9', 'high_vol_bearish', 1, 0.423, 0.373, NULL, 100.0, 13.0),
    ('MACD Crossover 12_26_9', 'neutral', 204, -0.004, -10.991, -0.039, 51.5, 14.5),
    ('MACD Crossover 12_26_9', 'low_vol_bullish', 1388, 0.002, -66.164, 0.025, 48.8, 12.3),
    ('MACD Crossover 12_26_9', 'low_vol_bearish', 1319, -0.001, -66.766, -0.006, 50.5, 12.3),
    
    ('EMA Crossover 7_35', 'high_vol_bearish', 1, 0.816, 0.766, NULL, 100.0, 34.0),
    ('EMA Crossover 7_35', 'neutral', 65, -0.006, -3.638, -0.064, 50.8, 17.3),
    ('EMA Crossover 7_35', 'low_vol_bearish', 580, 0.005, -25.866, 0.037, 49.5, 25.1),
    ('EMA Crossover 7_35', 'low_vol_bullish', 563, -0.006, -31.661, -0.040, 52.0, 23.9),
    
    ('RSI Threshold 11_40', 'neutral', 156, 0.010, -6.297, 0.113, 50.6, 24.0),
    ('RSI Threshold 11_40', 'low_vol_bullish', 122, -0.011, -7.418, -0.073, 52.5, 23.0),
    ('RSI Threshold 11_40', 'low_vol_bearish', 3308, 0.001, -161.179, 0.015, 50.0, 22.0)
) AS t(strategy_name, entry_regime, trade_count, avg_return_pct, net_return_pct, sharpe_ratio, win_rate, avg_duration_min);

-- HIGH VOLATILITY BEARISH REGIME
SELECT 
    '=== HIGH VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    ROUND(avg_return_pct, 3) as avg_return_pct,
    ROUND(net_return_pct, 2) as net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min
FROM strategy_results
WHERE entry_regime = 'high_vol_bearish'
ORDER BY COALESCE(sharpe_ratio, -999) DESC, net_return_pct DESC;

-- NEUTRAL REGIME
SELECT 
    '=== NEUTRAL REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    ROUND(avg_return_pct, 3) as avg_return_pct,
    ROUND(net_return_pct, 2) as net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min
FROM strategy_results
WHERE entry_regime = 'neutral'
ORDER BY sharpe_ratio DESC;

-- LOW VOLATILITY BULLISH REGIME
SELECT 
    '=== LOW VOLATILITY BULLISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    ROUND(avg_return_pct, 3) as avg_return_pct,
    ROUND(net_return_pct, 2) as net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min
FROM strategy_results
WHERE entry_regime = 'low_vol_bullish'
ORDER BY sharpe_ratio DESC;

-- LOW VOLATILITY BEARISH REGIME
SELECT 
    '=== LOW VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trade_count,
    ROUND(avg_return_pct, 3) as avg_return_pct,
    ROUND(net_return_pct, 2) as net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min
FROM strategy_results
WHERE entry_regime = 'low_vol_bearish'
ORDER BY sharpe_ratio DESC;

-- SHARPE RATIO RANKINGS ACROSS ALL REGIMES
SELECT 
    '=== TOP STRATEGIES BY SHARPE RATIO (ALL REGIMES) ===' as section;

SELECT 
    strategy_name,
    entry_regime,
    trade_count,
    sharpe_ratio,
    ROUND(net_return_pct, 2) as net_return_pct,
    ROUND(win_rate, 1) as win_rate_pct
FROM strategy_results
WHERE sharpe_ratio IS NOT NULL
ORDER BY sharpe_ratio DESC;

-- SUMMARY STATISTICS BY REGIME
SELECT 
    '=== SHARPE RATIO STATISTICS BY REGIME ===' as section;

SELECT 
    entry_regime,
    COUNT(DISTINCT strategy_name) as strategies,
    COUNT(CASE WHEN sharpe_ratio IS NOT NULL THEN 1 END) as with_sharpe,
    ROUND(MIN(sharpe_ratio), 3) as min_sharpe,
    ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
    ROUND(MAX(sharpe_ratio), 3) as max_sharpe,
    SUM(trade_count) as total_trades,
    ROUND(AVG(net_return_pct), 2) as avg_net_return_pct
FROM strategy_results
GROUP BY entry_regime
ORDER BY avg_sharpe DESC;

-- KEY INSIGHTS
SELECT 
    '=== KEY INSIGHTS ===' as section;

SELECT 
    'Best Sharpe in Neutral: RSI Threshold (0.113)' as insight
UNION ALL
SELECT 
    'Best Sharpe in Low Vol Bearish: EMA Crossover (0.037)' as insight
UNION ALL  
SELECT 
    'Best Sharpe in Low Vol Bullish: MACD Crossover (0.025)' as insight
UNION ALL
SELECT 
    'Worst Sharpe in Low Vol Bullish: RSI Threshold (-0.073)' as insight
UNION ALL
SELECT 
    'All strategies have negative returns after transaction costs' as insight;

DROP TABLE strategy_results;