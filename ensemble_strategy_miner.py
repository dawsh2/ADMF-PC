#!/usr/bin/env python3
"""
Ensemble strategy mining using SQL analytics.
Finds low-correlation strategy combinations for portfolio construction.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


class EnsembleStrategyMiner:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.con = duckdb.connect(str(self.workspace_path / "analytics.duckdb"))
    
    def find_ensemble_candidates(self, data_path: str, min_trades: int = 50):
        """Find strategies suitable for ensemble combination."""
        
        print("=== Ensemble Strategy Mining ===\n")
        
        # 1. Find profitable strategies with different characteristics
        print("1. Finding diverse profitable strategies...")
        
        diversity_query = f"""
        WITH strategy_characteristics AS (
            SELECT 
                s.strat,
                COUNT(*) as total_signals,
                SUM(CASE WHEN s.val = 1 THEN 1 ELSE 0 END) as long_signals,
                SUM(CASE WHEN s.val = -1 THEN 1 ELSE 0 END) as short_signals,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return_pct,
                -- Signal frequency (signals per 1000 bars)
                COUNT(*) * 1000.0 / (MAX(s.idx) - MIN(s.idx)) as signal_frequency,
                -- Strategy type from name
                CASE 
                    WHEN strat LIKE '%momentum%' THEN 'momentum'
                    WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                    WHEN strat LIKE '%rsi%' THEN 'rsi'
                    WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                    ELSE 'other'
                END as strategy_type
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
            GROUP BY s.strat
            HAVING COUNT(*) >= {min_trades}
        )
        SELECT 
            strategy_type,
            COUNT(*) as strategies,
            AVG(signal_frequency) as avg_frequency,
            AVG(avg_return_pct) as avg_return,
            MIN(avg_return_pct) as min_return,
            MAX(avg_return_pct) as max_return
        FROM strategy_characteristics
        WHERE avg_return_pct > 0  -- Profitable before costs
        GROUP BY strategy_type
        ORDER BY avg_return DESC
        """
        
        diversity_df = self.con.execute(diversity_query).df()
        print(diversity_df.to_string(index=False))
        
        # 2. Find low-correlation pairs
        print("\n\n2. Finding low-correlation strategy pairs...")
        
        correlation_query = f"""
        WITH signals AS (
            SELECT 
                strat,
                idx,
                val,
                -- Extract strategy type
                CASE 
                    WHEN strat LIKE '%momentum%' THEN 'momentum'
                    WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                    WHEN strat LIKE '%rsi%' THEN 'rsi'
                    WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                    ELSE 'other'
                END as strategy_type
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
            WHERE val != 0
        ),
        signal_arrays AS (
            SELECT 
                s1.strat as strat1,
                s1.strategy_type as type1,
                s2.strat as strat2,
                s2.strategy_type as type2,
                LIST(s1.idx ORDER BY s1.idx) as indices1,
                LIST(s2.idx ORDER BY s2.idx) as indices2
            FROM signals s1
            CROSS JOIN signals s2
            WHERE s1.strat < s2.strat  -- Avoid duplicates
                AND s1.strategy_type != s2.strategy_type  -- Different types
            GROUP BY s1.strat, s1.strategy_type, s2.strat, s2.strategy_type
        ),
        correlations AS (
            SELECT 
                strat1,
                type1,
                strat2,
                type2,
                LENGTH(indices1) as signals1,
                LENGTH(indices2) as signals2,
                LENGTH(LIST_INTERSECT(indices1, indices2)) as overlap,
                ROUND(LENGTH(LIST_INTERSECT(indices1, indices2)) * 100.0 / 
                      LEAST(LENGTH(indices1), LENGTH(indices2)), 2) as overlap_pct
            FROM signal_arrays
            WHERE LENGTH(indices1) >= {min_trades} 
                AND LENGTH(indices2) >= {min_trades}
        )
        SELECT 
            type1,
            type2,
            COUNT(*) as pairs,
            AVG(overlap_pct) as avg_overlap,
            MIN(overlap_pct) as min_overlap,
            MAX(overlap_pct) as max_overlap
        FROM correlations
        GROUP BY type1, type2
        ORDER BY avg_overlap
        """
        
        correlation_df = self.con.execute(correlation_query).df()
        print(correlation_df.to_string(index=False))
        
        # 3. Find best specific pairs
        print("\n\n3. Best low-correlation strategy pairs:")
        
        best_pairs_query = f"""
        WITH performance AS (
            SELECT 
                s.strat,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
            GROUP BY s.strat
            HAVING COUNT(*) >= {min_trades} AND AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) > 0
        ),
        signals AS (
            SELECT strat, LIST(idx ORDER BY idx) as indices
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
            WHERE val != 0
            GROUP BY strat
        ),
        pairs AS (
            SELECT 
                p1.strat as strat1,
                p2.strat as strat2,
                p1.avg_return as return1,
                p2.avg_return as return2,
                p1.trades as trades1,
                p2.trades as trades2,
                LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) * 100.0 / 
                    LEAST(LENGTH(s1.indices), LENGTH(s2.indices)) as overlap_pct
            FROM performance p1
            JOIN performance p2 ON p1.strat < p2.strat
            JOIN signals s1 ON p1.strat = s1.strat
            JOIN signals s2 ON p2.strat = s2.strat
            WHERE overlap_pct < 30  -- Low correlation threshold
        )
        SELECT 
            strat1,
            strat2,
            ROUND(return1, 4) as return1_pct,
            ROUND(return2, 4) as return2_pct,
            trades1,
            trades2,
            ROUND(overlap_pct, 1) as overlap_pct,
            ROUND((return1 + return2) / 2, 4) as ensemble_return
        FROM pairs
        ORDER BY ensemble_return DESC
        LIMIT 20
        """
        
        pairs_df = self.con.execute(best_pairs_query).df()
        print(pairs_df.to_string(index=False))
        
        # 4. Simulate ensemble performance
        print("\n\n4. Simulating ensemble portfolio performance:")
        
        # Get top 3 pairs
        if len(pairs_df) > 0:
            for i in range(min(3, len(pairs_df))):
                row = pairs_df.iloc[i]
                ensemble_query = f"""
                WITH signals1 AS (
                    SELECT idx, val FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
                    WHERE strat = '{row['strat1']}' AND val != 0
                ),
                signals2 AS (
                    SELECT idx, val FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
                    WHERE strat = '{row['strat2']}' AND val != 0
                ),
                combined_signals AS (
                    SELECT idx, val, 1 as weight FROM signals1
                    UNION ALL
                    SELECT idx, val, 1 as weight FROM signals2
                ),
                ensemble_trades AS (
                    SELECT 
                        idx,
                        SUM(val * weight) / SUM(weight) as ensemble_signal
                    FROM combined_signals
                    GROUP BY idx
                    HAVING ABS(ensemble_signal) >= 0.5  -- Majority vote
                )
                SELECT 
                    COUNT(*) as ensemble_trades,
                    AVG((m2.close - m1.close) / m1.close * 100 * SIGN(ensemble_signal)) as avg_return,
                    STDDEV((m2.close - m1.close) / m1.close * 100 * SIGN(ensemble_signal)) as volatility
                FROM ensemble_trades e
                JOIN read_parquet('{data_path}') m1 ON e.idx = m1.bar_index
                JOIN read_parquet('{data_path}') m2 ON e.idx + 1 = m2.bar_index
                """
                
                ensemble_result = self.con.execute(ensemble_query).df()
                print(f"\nEnsemble {i+1}: {row['strat1'][:30]}... + {row['strat2'][:30]}...")
                print(f"Individual returns: {row['return1_pct']}% + {row['return2_pct']}%")
                print(f"Overlap: {row['overlap_pct']}%")
                if len(ensemble_result) > 0:
                    print(f"Ensemble trades: {ensemble_result.iloc[0]['ensemble_trades']}")
                    print(f"Ensemble return: {ensemble_result.iloc[0]['avg_return']:.4f}%")
                    print(f"Sharpe: {ensemble_result.iloc[0]['avg_return'] / ensemble_result.iloc[0]['volatility']:.3f}")
    
    def analyze_voting_strategies(self, data_path: str):
        """Analyze different voting mechanisms for ensembles."""
        
        print("\n\n=== Voting Strategy Analysis ===")
        
        # Get top performing strategies
        top_query = f"""
        WITH performance AS (
            SELECT 
                s.strat,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
            GROUP BY s.strat
            HAVING COUNT(*) >= 100 AND AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) > 0
        )
        SELECT strat FROM performance ORDER BY avg_return DESC LIMIT 5
        """
        
        top_strategies = self.con.execute(top_query).df()['strat'].tolist()
        
        if len(top_strategies) >= 3:
            print(f"\nAnalyzing ensemble of top {len(top_strategies)} strategies")
            
            # Test different voting thresholds
            for threshold in [0.5, 0.6, 0.8, 1.0]:
                voting_query = f"""
                WITH all_signals AS (
                    SELECT idx, val 
                    FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
                    WHERE strat IN ({','.join([f"'{s}'" for s in top_strategies])})
                        AND val != 0
                ),
                voting AS (
                    SELECT 
                        idx,
                        AVG(val) as avg_signal,
                        COUNT(*) as votes
                    FROM all_signals
                    GROUP BY idx
                    HAVING ABS(AVG(val)) >= {threshold}
                )
                SELECT 
                    COUNT(*) as trades,
                    AVG((m2.close - m1.close) / m1.close * 100 * SIGN(avg_signal)) as avg_return
                FROM voting v
                JOIN read_parquet('{data_path}') m1 ON v.idx = m1.bar_index
                JOIN read_parquet('{data_path}') m2 ON v.idx + 1 = m2.bar_index
                """
                
                result = self.con.execute(voting_query).df()
                if len(result) > 0:
                    print(f"\nThreshold {threshold}: {result.iloc[0]['trades']} trades, "
                          f"{result.iloc[0]['avg_return']:.4f}% avg return")
    
    def close(self):
        self.con.close()


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ensemble_strategy_miner.py <workspace_path> <data_path>")
        sys.exit(1)
    
    miner = EnsembleStrategyMiner(sys.argv[1])
    
    try:
        miner.find_ensemble_candidates(sys.argv[2])
        miner.analyze_voting_strategies(sys.argv[2])
    finally:
        miner.close()
    
    print("\n\n=== Ensemble Strategy Recommendations ===")
    print("1. Combine momentum + mean reversion for all-weather performance")
    print("2. Use 30% overlap as maximum correlation threshold")
    print("3. Start with simple majority voting (>50% agreement)")
    print("4. Weight strategies by their individual Sharpe ratios")
    print("5. Consider time-of-day and volume filters for additional edge")


if __name__ == "__main__":
    main()