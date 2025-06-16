
-- Summary of what we learned about signal patterns
WITH signal_summary AS (
    SELECT 'MACD_CROSSOVER' as strategy, 'Alternating +1/-1 only' as pattern, 
           'Total signals ÷ 2' as trade_count_method
    UNION ALL
    SELECT 'RSI_THRESHOLD' as strategy, 'Alternating +1/-1 only' as pattern,
           'Total signals ÷ 2' as trade_count_method  
    UNION ALL
    SELECT 'WILLIAMS_R' as strategy, '+1,0,-1,0 pattern' as pattern,
           'Non-zero signals ÷ 2' as trade_count_method
    UNION ALL
    SELECT 'CCI_BANDS' as strategy, '+1,0,-1,0 pattern' as pattern,
           'Non-zero signals ÷ 2' as trade_count_method
)

SELECT * FROM signal_summary;
