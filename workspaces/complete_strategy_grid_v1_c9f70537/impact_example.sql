
-- Example: Impact on Sharpe ratio calculation
WITH example_calculation AS (
    SELECT 
        'neutral' as regime,
        100 as num_trades,
        0.5 as avg_return_pct,
        2.0 as std_return_pct,
        0.085 as total_time_wrong,
        0.029 as regime_time_correct
)

SELECT 
    regime,
    num_trades,
    -- WRONG calculation (using total time)
    ROUND(num_trades / total_time_wrong, 1) as trades_per_year_wrong,
    ROUND((avg_return_pct * num_trades / total_time_wrong) / 
          (std_return_pct * SQRT(num_trades / total_time_wrong)), 3) as sharpe_wrong,
    
    -- CORRECT calculation (using regime-specific time)
    ROUND(num_trades / regime_time_correct, 1) as trades_per_year_correct,
    ROUND((avg_return_pct * num_trades / regime_time_correct) / 
          (std_return_pct * SQRT(num_trades / regime_time_correct)), 3) as sharpe_correct
FROM example_calculation;
