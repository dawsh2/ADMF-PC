
-- Detailed analysis of one strategy with proper trade simulation
WITH signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_val,
        ROW_NUMBER() OVER (ORDER BY ts) as seq_num
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
),

-- Add regime information
signals_with_regime AS (
    SELECT 
        s.*,
        c.val as regime
    FROM signals s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
),

-- Calculate trade pairs (entry + exit = one complete trade)
trade_analysis AS (
    SELECT 
        regime,
        COUNT(*) as total_signals,
        COUNT(*) / 2.0 as complete_trades  -- Each trade = entry + exit
    FROM signals_with_regime
    GROUP BY regime
),

-- Get regime time distribution
regime_times AS (
    SELECT 
        regime,
        COUNT(*) / (252.0 * 390.0) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY regime
)

-- Show trading frequency analysis
SELECT 
    t.regime,
    t.complete_trades,
    r.regime_years,
    ROUND(t.complete_trades / r.regime_years, 1) as trades_per_year,
    ROUND(t.complete_trades / (r.regime_years * 252), 1) as trades_per_day
FROM trade_analysis t
JOIN regime_times r ON t.regime = r.regime
ORDER BY trades_per_day DESC;
