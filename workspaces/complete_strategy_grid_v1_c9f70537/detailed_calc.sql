
-- Detailed calculation for MACD strategy
WITH time_data AS (
    SELECT 297.0 as total_days, 297.0 / 365.25 as total_years  -- Mar 26 to Jan 17
),

regime_breakdown AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct,
        -- Calculate regime-specific time periods
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0 / 365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

macd_by_regime AS (
    SELECT 
        c.val as regime,
        COUNT(*) as total_signals
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    GROUP BY c.val
)

-- Show detailed breakdown
SELECT 
    r.regime,
    r.regime_minutes,
    ROUND(r.regime_pct, 1) as regime_pct,
    ROUND(r.regime_years, 3) as regime_years,
    m.total_signals,
    ROUND(m.total_signals / 2.0, 0) as complete_trades,
    ROUND((m.total_signals / 2.0) / r.regime_years, 1) as trades_per_year,
    ROUND((m.total_signals / 2.0) / (r.regime_years * 252), 2) as trades_per_trading_day
FROM regime_breakdown r
JOIN macd_by_regime m ON r.regime = m.regime
ORDER BY r.regime;
