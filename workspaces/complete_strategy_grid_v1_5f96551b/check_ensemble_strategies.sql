-- Check if ensemble strategies survive 0.5 bps transaction costs
PRAGMA memory_limit='3GB';
SET threads=4;

-- Define ensemble strategies from our DuckDBEnsemble configuration
WITH ensemble_strategies AS (
    SELECT strategy_type, strategy_name FROM (VALUES
        -- Low Vol Bullish strategies
        ('dema_crossover', 'dema_crossover_grid_19_35'),
        ('dema_crossover', 'dema_crossover_grid_7_35'),
        ('macd_crossover', 'macd_crossover_grid_12_35_9'),
        ('macd_crossover', 'macd_crossover_grid_15_35_7'),
        ('cci_threshold', 'cci_threshold_grid_11_-40'),
        ('pivot_channel_bounces', 'pivot_channel_bounces_grid_20_3_0.003'),
        
        -- Low Vol Bearish strategies  
        ('stochastic_crossover', 'stochastic_crossover_grid_27_5'),
        ('cci_threshold', 'cci_threshold_grid_11_-20'),
        ('ema_sma_crossover', 'ema_sma_crossover_grid_11_15'),
        ('keltner_breakout', 'keltner_breakout_grid_11_1.5'),
        ('rsi_bands', 'rsi_bands_grid_7_25_70'),
        ('pivot_channel_bounces', 'pivot_channel_bounces_grid_20_2_0.001'),
        
        -- Neutral strategies
        ('stochastic_rsi', 'stochastic_rsi_grid_21_21_15_80'),
        ('dema_crossover', 'dema_crossover_grid_19_15'),
        ('vortex_crossover', 'vortex_crossover_grid_27'),
        ('bollinger_breakout', 'bollinger_breakout_grid_11_2.5'),
        
        -- High Vol strategies
        ('keltner_breakout', 'keltner_breakout_grid_19_2.5'),
        ('bollinger_breakout', 'bollinger_breakout_grid_20_2.0'),
        ('atr_channel_breakout', 'atr_channel_breakout_grid_14_20_2.0'),
        
        -- Mean reversion strategies we added
        ('bollinger_mean_reversion', 'bollinger_mean_reversion_grid_20_2.0'),
        ('keltner_mean_reversion', 'keltner_mean_reversion_grid_20_2.0')
    ) AS t(strategy_type, strategy_name)
),
cost_impact AS (
    SELECT 
        strategy_name,
        strategy_type,
        current_regime,
        sharpe_ratio as gross_sharpe,
        total_return,
        win_rate,
        trading_days,
        
        -- Net daily return after 0.5 bps costs
        (total_return / NULLIF(trading_days, 0)) - 
        CASE 
            WHEN trading_days > 0 
            THEN (ROUND(trading_days * 2.0) * 0.5 / 10000) / trading_days
            ELSE 0
        END as net_daily_return_pct
    FROM analytics.strategy_scores
    WHERE sharpe_ratio IS NOT NULL 
        AND total_return IS NOT NULL
        AND trading_days >= 20
),
ensemble_cost_check AS (
    SELECT 
        e.strategy_type,
        e.strategy_name,
        c.current_regime,
        c.gross_sharpe,
        ROUND(c.net_daily_return_pct * 10000, 2) as net_daily_return_bps,
        CASE 
            WHEN c.net_daily_return_pct > 0 THEN 'SURVIVED'
            WHEN c.net_daily_return_pct IS NOT NULL THEN 'ELIMINATED_BY_COSTS'
            ELSE 'NOT_FOUND_IN_ANALYSIS'
        END as cost_status
    FROM ensemble_strategies e
    LEFT JOIN cost_impact c ON e.strategy_type = c.strategy_type 
        AND e.strategy_name = c.strategy_name
)
SELECT 
    strategy_type,
    strategy_name,
    current_regime,
    gross_sharpe,
    net_daily_return_bps,
    cost_status
FROM ensemble_cost_check
ORDER BY 
    CASE cost_status 
        WHEN 'SURVIVED' THEN 1
        WHEN 'ELIMINATED_BY_COSTS' THEN 2  
        ELSE 3
    END,
    net_daily_return_bps DESC NULLS LAST;

-- Summary
SELECT 
    '=== ENSEMBLE COST IMPACT SUMMARY ===' as summary;
    
SELECT 
    cost_status,
    COUNT(*) as strategy_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ensemble_cost_check), 1) as percentage
FROM ensemble_cost_check
GROUP BY cost_status
ORDER BY strategy_count DESC;