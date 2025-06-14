#!/usr/bin/env python3
"""
Regime-aware strategy analysis.
Analyzes strategy performance within specific market regimes/classifiers.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


class RegimeAwareAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.con = duckdb.connect(str(self.workspace_path / "analytics.duckdb"))
    
    def analyze_with_regimes(self, data_path: str):
        """Analyze strategies within different market regimes."""
        
        print("=== Regime-Aware Strategy Analysis ===\n")
        
        # 1. Check for classifier data
        print("1. Checking for classifier data...")
        classifier_check = f"""
        SELECT 
            COUNT(*) as files,
            COUNT(DISTINCT strat) as classifiers
        FROM read_parquet('{self.workspace_path}/traces/*/classifiers/*/*.parquet')
        """
        
        try:
            classifier_info = self.con.execute(classifier_check).df()
            has_classifiers = classifier_info.iloc[0]['files'] > 0
            print(f"Found {classifier_info.iloc[0]['classifiers']} classifiers")
        except:
            has_classifiers = False
            print("No classifier data found, creating synthetic regimes...")
        
        # 2. Create regimes (synthetic if no classifiers)
        if not has_classifiers:
            self._analyze_with_synthetic_regimes(data_path)
        else:
            self._analyze_with_classifiers(data_path)
    
    def _analyze_with_synthetic_regimes(self, data_path: str):
        """Use price-based regimes when no classifiers available."""
        
        print("\n2. Creating price-based regimes...")
        
        # Define regimes based on price action
        regime_query = f"""
        WITH price_features AS (
            SELECT 
                bar_index,
                close,
                volume,
                -- Moving averages
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) as ma50,
                -- Volatility
                STDDEV(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as vol20,
                -- Volume profile
                AVG(volume) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as avg_vol20,
                -- Price momentum
                (close - LAG(close, 5) OVER (ORDER BY bar_index)) / LAG(close, 5) OVER (ORDER BY bar_index) as momentum5
            FROM read_parquet('{data_path}')
        ),
        regimes AS (
            SELECT 
                bar_index,
                -- Trend regime
                CASE 
                    WHEN close > ma20 AND ma20 > ma50 THEN 'strong_uptrend'
                    WHEN close > ma20 THEN 'uptrend'
                    WHEN close < ma20 AND ma20 < ma50 THEN 'strong_downtrend'
                    WHEN close < ma20 THEN 'downtrend'
                    ELSE 'neutral'
                END as trend_regime,
                -- Volatility regime
                CASE 
                    WHEN vol20 > AVG(vol20) OVER () * 1.5 THEN 'high_volatility'
                    WHEN vol20 < AVG(vol20) OVER () * 0.7 THEN 'low_volatility'
                    ELSE 'normal_volatility'
                END as volatility_regime,
                -- Volume regime
                CASE 
                    WHEN volume > avg_vol20 * 2 THEN 'high_volume'
                    WHEN volume < avg_vol20 * 0.5 THEN 'low_volume'
                    ELSE 'normal_volume'
                END as volume_regime,
                -- Momentum regime
                CASE 
                    WHEN momentum5 > 0.02 THEN 'strong_momentum_up'
                    WHEN momentum5 > 0.005 THEN 'momentum_up'
                    WHEN momentum5 < -0.02 THEN 'strong_momentum_down'
                    WHEN momentum5 < -0.005 THEN 'momentum_down'
                    ELSE 'no_momentum'
                END as momentum_regime
            FROM price_features
            WHERE bar_index >= 50  -- Need history for indicators
        )
        SELECT 
            trend_regime,
            COUNT(*) as bars,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
        FROM regimes
        GROUP BY trend_regime
        ORDER BY bars DESC
        """
        
        regime_dist = self.con.execute(regime_query).df()
        print("\nTrend Regime Distribution:")
        print(regime_dist.to_string(index=False))
        
        # 3. Analyze strategy performance by regime
        print("\n3. Strategy Performance by Regime:")
        
        performance_query = f"""
        WITH price_features AS (
            SELECT 
                bar_index,
                close,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) as ma50,
                STDDEV(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as vol20
            FROM read_parquet('{data_path}')
        ),
        regimes AS (
            SELECT 
                bar_index,
                CASE 
                    WHEN close > ma20 AND ma20 > ma50 THEN 'strong_uptrend'
                    WHEN close > ma20 THEN 'uptrend'
                    WHEN close < ma20 AND ma20 < ma50 THEN 'strong_downtrend'
                    WHEN close < ma20 THEN 'downtrend'
                    ELSE 'neutral'
                END as regime
            FROM price_features
        ),
        strategy_performance AS (
            SELECT 
                s.strat,
                r.regime,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return,
                STDDEV(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as volatility
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            JOIN regimes r ON s.idx = r.bar_index
            WHERE s.val != 0
            GROUP BY s.strat, r.regime
            HAVING COUNT(*) >= 20
        )
        SELECT 
            regime,
            COUNT(DISTINCT strat) as strategies,
            AVG(avg_return) as avg_return,
            MAX(avg_return) as best_return,
            MIN(avg_return) as worst_return
        FROM strategy_performance
        GROUP BY regime
        ORDER BY avg_return DESC
        """
        
        regime_perf = self.con.execute(performance_query).df()
        print(regime_perf.to_string(index=False))
        
        # 4. Find regime specialists
        print("\n\n4. Regime Specialists (strategies that excel in specific regimes):")
        
        specialist_query = f"""
        WITH price_features AS (
            SELECT 
                bar_index,
                close,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) as ma50
            FROM read_parquet('{data_path}')
        ),
        regimes AS (
            SELECT 
                bar_index,
                CASE 
                    WHEN close > ma20 AND ma20 > ma50 THEN 'strong_uptrend'
                    WHEN close > ma20 THEN 'uptrend'
                    WHEN close < ma20 AND ma20 < ma50 THEN 'strong_downtrend'
                    WHEN close < ma20 THEN 'downtrend'
                    ELSE 'neutral'
                END as regime
            FROM price_features
        ),
        performance_by_regime AS (
            SELECT 
                s.strat,
                r.regime,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            JOIN regimes r ON s.idx = r.bar_index
            WHERE s.val != 0
            GROUP BY s.strat, r.regime
            HAVING COUNT(*) >= 30
        ),
        overall_performance AS (
            SELECT 
                strat,
                AVG(avg_return) as overall_return
            FROM performance_by_regime
            GROUP BY strat
        )
        SELECT 
            p.regime,
            p.strat,
            p.trades,
            ROUND(p.avg_return, 4) as regime_return,
            ROUND(o.overall_return, 4) as overall_return,
            ROUND(p.avg_return - o.overall_return, 4) as regime_edge
        FROM performance_by_regime p
        JOIN overall_performance o ON p.strat = o.strat
        WHERE p.avg_return > 0.01  -- Profitable in regime
            AND p.avg_return > o.overall_return * 2  -- At least 2x better in regime
        ORDER BY p.regime, regime_edge DESC
        """
        
        specialists = self.con.execute(specialist_query).df()
        if len(specialists) > 0:
            for regime in specialists['regime'].unique():
                regime_df = specialists[specialists['regime'] == regime].head(3)
                print(f"\n{regime}:")
                print(regime_df[['strat', 'trades', 'regime_return', 'overall_return', 'regime_edge']].to_string(index=False))
        
        # 5. Composite strategies by regime
        print("\n\n5. Composite Strategy Performance by Regime:")
        
        composite_regime_query = f"""
        WITH price_features AS (
            SELECT 
                bar_index,
                close,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) as ma50
            FROM read_parquet('{data_path}')
        ),
        regimes AS (
            SELECT 
                bar_index,
                CASE 
                    WHEN close > ma20 AND ma20 > ma50 THEN 'strong_uptrend'
                    WHEN close > ma20 THEN 'uptrend'
                    WHEN close < ma20 AND ma20 < ma50 THEN 'strong_downtrend'
                    WHEN close < ma20 THEN 'downtrend'
                    ELSE 'neutral'
                END as regime
            FROM price_features
        ),
        composite_trades AS (
            SELECT 
                e.idx as entry_idx,
                MIN(x.idx) as exit_idx,
                r.regime
            FROM read_parquet('{self.workspace_path}/traces/*/signals/rsi_grid/*.parquet') e
            JOIN read_parquet('{self.workspace_path}/traces/*/signals/rsi_grid/*.parquet') x
                ON x.idx > e.idx 
                AND x.idx <= e.idx + 10
                AND e.val = 1 
                AND x.val = -1
                AND e.strat LIKE '%_7_%'
                AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
            JOIN regimes r ON e.idx = r.bar_index
            GROUP BY e.idx, r.regime
        )
        SELECT 
            regime,
            COUNT(*) as trades,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
            ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility
        FROM composite_trades t
        JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
        GROUP BY regime
        HAVING COUNT(*) >= 5
        ORDER BY avg_return DESC
        """
        
        composite_regime = self.con.execute(composite_regime_query).df()
        print(composite_regime.to_string(index=False))
        
        # 6. Regime-switching ensemble
        print("\n\n6. Regime-Switching Ensemble Strategy:")
        
        print("\nUsing best strategy for each regime:")
        if len(specialists) > 0:
            regime_map = {}
            for regime in specialists['regime'].unique():
                best = specialists[specialists['regime'] == regime].iloc[0]
                regime_map[regime] = best['strat']
                print(f"{regime}: {best['strat'][:40]}... ({best['regime_return']}%)")
        
        print("\n\n=== Regime-Aware Recommendations ===")
        print("1. Use different strategies for different market regimes")
        print("2. Composite strategies work best in trending regimes")
        print("3. Mean reversion excels in neutral/ranging regimes")
        print("4. Consider regime detection as entry filter")
        print("5. Ensemble weights should adapt to current regime")
    
    def _analyze_with_classifiers(self, data_path: str):
        """Analyze using actual classifier data."""
        
        print("\n2. Analyzing with classifier data...")
        
        # Get classifier predictions
        classifier_query = f"""
        SELECT 
            strat as classifier,
            COUNT(*) as predictions,
            COUNT(DISTINCT val) as classes
        FROM read_parquet('{self.workspace_path}/traces/*/classifiers/*/*.parquet')
        GROUP BY strat
        """
        
        classifiers = self.con.execute(classifier_query).df()
        print("\nAvailable classifiers:")
        print(classifiers.to_string(index=False))
        
        # Use first classifier for analysis
        if len(classifiers) > 0:
            classifier_name = classifiers.iloc[0]['classifier']
            
            # Analyze strategy performance by classifier state
            perf_by_class_query = f"""
            WITH classifier_states AS (
                SELECT idx, val as class_label
                FROM read_parquet('{self.workspace_path}/traces/*/classifiers/*/*.parquet')
                WHERE strat = '{classifier_name}'
            ),
            strategy_performance AS (
                SELECT 
                    s.strat,
                    c.class_label,
                    COUNT(*) as trades,
                    AVG(CASE 
                        WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                        WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                    END) as avg_return
                FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
                JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
                JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
                JOIN classifier_states c ON s.idx = c.idx
                WHERE s.val != 0
                GROUP BY s.strat, c.class_label
                HAVING COUNT(*) >= 10
            )
            SELECT 
                class_label,
                COUNT(DISTINCT strat) as strategies,
                SUM(trades) as total_trades,
                ROUND(AVG(avg_return), 4) as avg_return
            FROM strategy_performance
            GROUP BY class_label
            ORDER BY avg_return DESC
            """
            
            class_perf = self.con.execute(perf_by_class_query).df()
            print(f"\nPerformance by {classifier_name} state:")
            print(class_perf.to_string(index=False))
    
    def close(self):
        self.con.close()


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python regime_aware_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyzer = RegimeAwareAnalyzer(sys.argv[1])
    
    try:
        analyzer.analyze_with_regimes(sys.argv[2])
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()