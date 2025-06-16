-- Calculate performance statistics by regime for any strategy
-- Input: Requires trade returns from 01_calculate_trade_returns.sql
-- Output: Mean/std returns, trade counts, Sharpe ratios by regime

WITH regime_times AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

trade_stats AS (
    SELECT 
        regime,
        COUNT(*) as num_trades,
        AVG(return_decimal) as mean_return,
        STDDEV(return_decimal) as std_return,
        AVG(return_bps) as mean_return_bps,
        STDDEV(return_bps) as std_return_bps,
        MIN(return_bps) as min_return_bps,
        MAX(return_bps) as max_return_bps
    FROM (${trade_returns_query}) -- This will be replaced with actual trade returns
    GROUP BY regime
)

SELECT 
    '${strategy_name}' as strategy,
    s.regime,
    s.num_trades,
    ROUND(r.regime_years, 3) as regime_years,
    ROUND(s.num_trades / (r.regime_years * 252), 1) as trades_per_day,
    ROUND(s.mean_return * 100, 4) as mean_return_pct,
    ROUND(s.std_return * 100, 4) as std_return_pct,
    ROUND(s.mean_return_bps, 2) as mean_return_bps,
    ROUND(s.std_return_bps, 2) as std_return_bps,
    ROUND(s.min_return_bps, 2) as min_return_bps,
    ROUND(s.max_return_bps, 2) as max_return_bps,
    ROUND(s.mean_return / s.std_return, 6) as raw_sharpe,
    ROUND(s.num_trades / r.regime_years, 0) as trades_per_year,
    ROUND(
        (s.mean_return / s.std_return) * SQRT(s.num_trades / r.regime_years), 4
    ) as annualized_sharpe
FROM trade_stats s
JOIN regime_times r ON s.regime = r.regime
ORDER BY annualized_sharpe DESC;