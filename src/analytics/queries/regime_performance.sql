-- Analyze strategy performance across different market regimes
WITH market_metrics AS (
    -- Calculate rolling market metrics for regime detection
    SELECT 
        ts,
        -- Assuming we have market data joined or available
        close_price,
        high_price,
        low_price,
        volume,
        -- 20-period rolling volatility
        STDDEV(close_price) OVER (ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20,
        -- 50-period rolling volatility for comparison
        STDDEV(close_price) OVER (ORDER BY ts ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as volatility_50,
        -- Price momentum
        (close_price / LAG(close_price, 20) OVER (ORDER BY ts) - 1) as momentum_20,
        -- Volume metrics
        AVG(volume) OVER (ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20
    FROM market_data
    WHERE symbol = '{symbol}'
),
regime_classification AS (
    -- Classify each period into a regime
    SELECT 
        ts,
        CASE 
            WHEN volatility_20 < PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY volatility_20) OVER () THEN 'low_vol'
            WHEN volatility_20 > PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY volatility_20) OVER () THEN 'high_vol'
            ELSE 'medium_vol'
        END as volatility_regime,
        CASE
            WHEN momentum_20 > 0.05 THEN 'strong_uptrend'
            WHEN momentum_20 > 0.02 THEN 'uptrend'
            WHEN momentum_20 < -0.05 THEN 'strong_downtrend'
            WHEN momentum_20 < -0.02 THEN 'downtrend'
            ELSE 'sideways'
        END as trend_regime,
        CASE
            WHEN volume > avg_volume_20 * 1.5 THEN 'high_volume'
            WHEN volume < avg_volume_20 * 0.5 THEN 'low_volume'
            ELSE 'normal_volume'
        END as volume_regime
    FROM market_metrics
),
strategy_regime_performance AS (
    -- Join strategy signals with regime data
    SELECT 
        s.strategy_hash,
        r.volatility_regime,
        r.trend_regime,
        r.volume_regime,
        COUNT(*) as signals_in_regime,
        SUM(CASE WHEN s.val != 0 THEN 1 ELSE 0 END) as active_signals,
        -- You would calculate actual returns here with price data
        AVG(s.val) as avg_signal_strength
    FROM signals s
    JOIN regime_classification r ON DATE_TRUNC('hour', s.ts) = DATE_TRUNC('hour', r.ts)
    WHERE s.strategy_hash IN (SELECT strategy_hash FROM strategies WHERE sharpe_ratio > {min_sharpe})
    GROUP BY s.strategy_hash, r.volatility_regime, r.trend_regime, r.volume_regime
)
SELECT 
    srp.*,
    st.strategy_type,
    st.sharpe_ratio,
    st.total_return,
    -- Calculate regime-specific metrics
    srp.active_signals::FLOAT / NULLIF(srp.signals_in_regime, 0) as activity_rate,
    -- Rank within regime
    RANK() OVER (PARTITION BY srp.volatility_regime ORDER BY st.sharpe_ratio DESC) as rank_in_vol_regime,
    RANK() OVER (PARTITION BY srp.trend_regime ORDER BY st.sharpe_ratio DESC) as rank_in_trend_regime
FROM strategy_regime_performance srp
JOIN strategies st ON srp.strategy_hash = st.strategy_hash
ORDER BY st.sharpe_ratio DESC, srp.volatility_regime, srp.trend_regime