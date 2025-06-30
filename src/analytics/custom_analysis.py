"""
Custom analysis functions that extend the interactive framework.

These are examples of domain-specific analysis patterns you might want to save.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.analytics.interactive import BacktestRun, AnalysisWorkspace


def find_mean_reversion_patterns(run: BacktestRun) -> pd.DataFrame:
    """Find mean reversion strategies that work well in ranging markets."""
    return run.query("""
        WITH mean_reversion AS (
            SELECT * FROM strategies
            WHERE strategy_type IN ('mean_reversion', 'rsi', 'bollinger')
        ),
        signal_patterns AS (
            SELECT 
                s.strategy_hash,
                COUNT(*) as total_signals,
                AVG(CASE WHEN s.val > 0 THEN 1 ELSE 0 END) as long_ratio,
                -- Measure signal "choppiness" - frequent direction changes
                SUM(CASE WHEN s.val * LAG(s.val) OVER (PARTITION BY s.strategy_hash ORDER BY ts) < 0 THEN 1 ELSE 0 END) as direction_changes
            FROM signals s
            JOIN mean_reversion mr ON s.strategy_hash = mr.strategy_hash
            WHERE s.val != 0
            GROUP BY s.strategy_hash
        )
        SELECT 
            mr.*,
            sp.total_signals,
            sp.long_ratio,
            sp.direction_changes::FLOAT / sp.total_signals as reversal_frequency
        FROM mean_reversion mr
        JOIN signal_patterns sp ON mr.strategy_hash = sp.strategy_hash
        WHERE mr.sharpe_ratio > 1.0
        ORDER BY reversal_frequency DESC
    """)


def find_trend_following_patterns(run: BacktestRun) -> pd.DataFrame:
    """Find trend following strategies with good directional persistence."""
    return run.query("""
        WITH trend_strategies AS (
            SELECT * FROM strategies
            WHERE strategy_type IN ('momentum', 'trend_following', 'breakout')
        ),
        signal_persistence AS (
            SELECT 
                s.strategy_hash,
                COUNT(*) as total_signals,
                -- Average signal duration
                AVG(
                    CASE WHEN s.val != LAG(s.val) OVER (PARTITION BY s.strategy_hash ORDER BY ts) 
                    THEN 1 ELSE 0 END
                ) as signal_changes,
                -- Longest streak
                MAX(
                    SUM(CASE WHEN s.val = LAG(s.val) OVER (PARTITION BY s.strategy_hash ORDER BY ts) THEN 1 ELSE 0 END) 
                    OVER (PARTITION BY s.strategy_hash ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                ) as max_streak
            FROM signals s
            JOIN trend_strategies ts ON s.strategy_hash = ts.strategy_hash
            WHERE s.val != 0
            GROUP BY s.strategy_hash
        )
        SELECT 
            ts.*,
            sp.total_signals,
            1 - sp.signal_changes as persistence_score,
            sp.max_streak
        FROM trend_strategies ts
        JOIN signal_persistence sp ON ts.strategy_hash = sp.strategy_hash
        WHERE ts.sharpe_ratio > 1.0
        ORDER BY persistence_score DESC
    """)


def analyze_strategy_pairs(run: BacktestRun, threshold: float = 0.3) -> pd.DataFrame:
    """Find strategy pairs with complementary trading patterns."""
    return run.query(f"""
        WITH top_strategies AS (
            SELECT strategy_hash, strategy_type, sharpe_ratio
            FROM strategies
            WHERE sharpe_ratio > 1.5
            LIMIT 50
        ),
        signal_overlaps AS (
            SELECT 
                s1.strategy_hash as hash1,
                s2.strategy_hash as hash2,
                COUNT(DISTINCT DATE_TRUNC('hour', s1.ts)) as overlap_hours,
                -- How often they agree on direction
                SUM(CASE WHEN SIGN(s1.val) = SIGN(s2.val) THEN 1 ELSE 0 END)::FLOAT / 
                    COUNT(*) as direction_agreement
            FROM signals s1
            JOIN signals s2 
                ON DATE_TRUNC('minute', s1.ts) = DATE_TRUNC('minute', s2.ts)
                AND s1.strategy_hash < s2.strategy_hash
            WHERE s1.val != 0 AND s2.val != 0
            AND s1.strategy_hash IN (SELECT strategy_hash FROM top_strategies)
            AND s2.strategy_hash IN (SELECT strategy_hash FROM top_strategies)
            GROUP BY s1.strategy_hash, s2.strategy_hash
        )
        SELECT 
            t1.strategy_type as type1,
            t2.strategy_type as type2,
            so.direction_agreement,
            so.overlap_hours,
            t1.sharpe_ratio + t2.sharpe_ratio as combined_sharpe
        FROM signal_overlaps so
        JOIN top_strategies t1 ON so.hash1 = t1.strategy_hash
        JOIN top_strategies t2 ON so.hash2 = t2.strategy_hash
        WHERE so.direction_agreement < {threshold}  -- Low agreement = complementary
        ORDER BY combined_sharpe DESC
        LIMIT 20
    """)


def find_time_diversified_strategies(run: BacktestRun) -> Dict[str, pd.DataFrame]:
    """Find strategies that trade at different times of day."""
    
    # Get hourly activity for each strategy
    hourly_activity = run.query("""
        WITH hourly_signals AS (
            SELECT 
                strategy_hash,
                EXTRACT(HOUR FROM ts) as hour,
                COUNT(*) as signals
            FROM signals
            WHERE val != 0
            GROUP BY strategy_hash, hour
        ),
        strategy_profiles AS (
            SELECT 
                hs.strategy_hash,
                s.strategy_type,
                s.sharpe_ratio,
                hs.hour,
                hs.signals,
                hs.signals::FLOAT / SUM(hs.signals) OVER (PARTITION BY hs.strategy_hash) as hour_weight
            FROM hourly_signals hs
            JOIN strategies s ON hs.strategy_hash = s.strategy_hash
            WHERE s.sharpe_ratio > 1.0
        )
        SELECT * FROM strategy_profiles
        ORDER BY strategy_hash, hour
    """)
    
    # Group strategies by their peak trading hours
    morning_traders = hourly_activity[
        (hourly_activity['hour'] >= 9) & (hourly_activity['hour'] <= 12)
    ].groupby('strategy_hash').agg({'hour_weight': 'sum', 'sharpe_ratio': 'first'})
    morning_traders = morning_traders[morning_traders['hour_weight'] > 0.5]
    
    afternoon_traders = hourly_activity[
        (hourly_activity['hour'] >= 13) & (hourly_activity['hour'] <= 16)
    ].groupby('strategy_hash').agg({'hour_weight': 'sum', 'sharpe_ratio': 'first'})
    afternoon_traders = afternoon_traders[afternoon_traders['hour_weight'] > 0.5]
    
    return {
        'morning': morning_traders.sort_values('sharpe_ratio', ascending=False),
        'afternoon': afternoon_traders.sort_values('sharpe_ratio', ascending=False),
        'hourly_profile': hourly_activity
    }


def optimize_ensemble_weights(workspace: AnalysisWorkspace, run: BacktestRun, 
                            strategies: List[str]) -> Dict[str, float]:
    """
    Optimize portfolio weights using mean-variance optimization.
    
    This is a simplified version - in practice you'd want more sophisticated
    optimization with constraints, transaction costs, etc.
    """
    from scipy.optimize import minimize
    
    # Get returns for each strategy
    returns_data = {}
    for strategy_hash in strategies:
        returns_query = f"""
            SELECT 
                ts,
                val as signal,
                LAG(val) OVER (ORDER BY ts) as prev_signal
            FROM signals
            WHERE strategy_hash = '{strategy_hash}'
            ORDER BY ts
        """
        signals = run.query(returns_query)
        # Simplified return calculation - you'd use actual price data in practice
        returns_data[strategy_hash] = signals['signal'].diff().dropna()
    
    # Create returns matrix
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov()
    expected_returns = returns_df.mean()
    
    # Optimization
    n_assets = len(strategies)
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    def portfolio_return(weights):
        return expected_returns @ weights
    
    # Maximize Sharpe ratio
    def negative_sharpe(weights):
        ret = portfolio_return(weights)
        vol = np.sqrt(portfolio_variance(weights))
        return -ret / vol if vol > 0 else 0
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
    ]
    
    # Bounds (0 <= weight <= 0.4 for each asset)
    bounds = tuple((0, 0.4) for _ in range(n_assets))
    
    # Initial guess
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(negative_sharpe, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return dict(zip(strategies, result.x))
    else:
        # Fall back to equal weights
        return dict(zip(strategies, [1/n_assets] * n_assets))


def find_regime_adaptive_strategies(run: BacktestRun) -> pd.DataFrame:
    """Find strategies that adapt well to different market regimes."""
    return run.query("""
        WITH regime_performance AS (
            -- Simplified regime detection based on volatility
            SELECT 
                s.strategy_hash,
                s.ts,
                s.val,
                st.sharpe_ratio,
                -- Rolling 20-period volatility
                STDDEV(s.val) OVER (
                    PARTITION BY s.strategy_hash 
                    ORDER BY s.ts 
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as rolling_vol
            FROM signals s
            JOIN strategies st ON s.strategy_hash = st.strategy_hash
            WHERE s.val != 0
        ),
        regime_stats AS (
            SELECT 
                strategy_hash,
                CASE 
                    WHEN rolling_vol < PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY rolling_vol) OVER () THEN 'low_vol'
                    WHEN rolling_vol > PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY rolling_vol) OVER () THEN 'high_vol'
                    ELSE 'medium_vol'
                END as regime,
                COUNT(*) as signals_in_regime,
                sharpe_ratio
            FROM regime_performance
            WHERE rolling_vol IS NOT NULL
            GROUP BY strategy_hash, regime, sharpe_ratio
        ),
        regime_consistency AS (
            SELECT 
                strategy_hash,
                MAX(sharpe_ratio) as overall_sharpe,
                COUNT(DISTINCT regime) as regimes_traded,
                MIN(signals_in_regime) as min_signals_regime,
                STDDEV(signals_in_regime) as signal_variance
            FROM regime_stats
            GROUP BY strategy_hash
            HAVING COUNT(DISTINCT regime) = 3  -- Trades in all regimes
        )
        SELECT 
            rc.*,
            s.strategy_type,
            -- Consistency score: high signals in all regimes, low variance
            rc.min_signals_regime / NULLIF(rc.signal_variance, 0) as consistency_score
        FROM regime_consistency rc
        JOIN strategies s ON rc.strategy_hash = s.strategy_hash
        WHERE rc.overall_sharpe > 1.0
        ORDER BY consistency_score DESC
    """)


# Example usage in notebook:
"""
# Load these custom functions
from src.analytics.custom_analysis import *

# Find mean reversion patterns
mean_rev = find_mean_reversion_patterns(run)
print("Top Mean Reversion Strategies:")
print(mean_rev[['strategy_type', 'sharpe_ratio', 'reversal_frequency']].head())

# Find complementary pairs
pairs = analyze_strategy_pairs(run, threshold=0.3)
print("\nComplementary Strategy Pairs:")
print(pairs.head())

# Time diversification
time_div = find_time_diversified_strategies(run)
print("\nMorning Traders:")
print(time_div['morning'].head())
print("\nAfternoon Traders:")
print(time_div['afternoon'].head())

# Find adaptive strategies
adaptive = find_regime_adaptive_strategies(run)
print("\nRegime-Adaptive Strategies:")
print(adaptive[['strategy_type', 'overall_sharpe', 'consistency_score']].head())
"""