#!/usr/bin/env python3
"""
Advanced pattern mining with regime filtering and composite strategies.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


class AdvancedPatternMiner:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.con = duckdb.connect(str(self.workspace_path / "analytics.duckdb"))
    
    def analyze_composite_strategies(self, data_path: str):
        """Analyze entry/exit combinations discovered by pattern mining."""
        
        print("=== Composite Strategy Analysis ===\n")
        
        # Test the top pattern: RSI fast entry, RSI slow exit
        query = f"""
        WITH entry_signals AS (
            SELECT 
                idx as entry_idx,
                strat as entry_strategy
            FROM read_parquet('{self.workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE val = 1 AND strat LIKE 'SPY_rsi_grid_7_%'
        ),
        exit_signals AS (
            SELECT 
                idx as exit_idx,
                strat as exit_strategy
            FROM read_parquet('{self.workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE val = -1 AND (strat LIKE 'SPY_rsi_grid_14_%' OR strat LIKE 'SPY_rsi_grid_21_%')
        ),
        composite_trades AS (
            SELECT 
                e.entry_idx,
                MIN(x.exit_idx) as exit_idx,
                e.entry_strategy,
                FIRST(x.exit_strategy) as exit_strategy
            FROM entry_signals e
            JOIN exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 10
            GROUP BY e.entry_idx, e.entry_strategy
        )
        SELECT 
            COUNT(*) as trades,
            ROUND(AVG(exit_idx - entry_idx), 1) as avg_holding,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
            ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100) / 
                  NULLIF(STDDEV((m2.close - m1.close) / m1.close * 100), 0), 3) as sharpe
        FROM composite_trades ct
        JOIN read_parquet('{data_path}') m1 ON ct.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON ct.exit_idx = m2.bar_index
        """
        
        result = self.con.execute(query).df()
        print("1. RSI Fast Entry → RSI Slow Exit:")
        print(result.to_string(index=False))
        
        # Test mean reversion entry, RSI exit
        query2 = f"""
        WITH entry_signals AS (
            SELECT idx as entry_idx
            FROM read_parquet('{self.workspace_path}/traces/SPY_1m/signals/mean_reversion_grid/*.parquet')
            WHERE val = 1
        ),
        exit_signals AS (
            SELECT idx as exit_idx
            FROM read_parquet('{self.workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE val = -1 AND (strat LIKE 'SPY_rsi_grid_14_%' OR strat LIKE 'SPY_rsi_grid_21_%')
        ),
        composite_trades AS (
            SELECT 
                e.entry_idx,
                MIN(x.exit_idx) as exit_idx
            FROM entry_signals e
            JOIN exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 10
            GROUP BY e.entry_idx
        )
        SELECT 
            COUNT(*) as trades,
            ROUND(AVG(exit_idx - entry_idx), 1) as avg_holding,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return
        FROM composite_trades ct
        JOIN read_parquet('{data_path}') m1 ON ct.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON ct.exit_idx = m2.bar_index
        """
        
        result2 = self.con.execute(query2).df()
        print("\n2. Mean Reversion Entry → RSI Slow Exit:")
        print(result2.to_string(index=False))
    
    def analyze_regime_performance(self, data_path: str):
        """Analyze strategy performance by market regime."""
        
        print("\n\n=== Regime-Based Performance Analysis ===\n")
        
        # First, check if we have classifier data
        classifier_files = list(self.workspace_path.glob("traces/*/classifiers/*/*.parquet"))
        
        if not classifier_files:
            print("No classifier data found. Generating synthetic regimes based on price action...")
            
            # Create synthetic regimes based on price movement
            regime_query = f"""
            WITH price_changes AS (
                SELECT 
                    bar_index,
                    close,
                    AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                    STDDEV(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as std20
                FROM read_parquet('{data_path}')
            ),
            regimes AS (
                SELECT 
                    bar_index,
                    CASE 
                        WHEN close > ma20 + std20 THEN 'strong_trend_up'
                        WHEN close > ma20 THEN 'trend_up'
                        WHEN close < ma20 - std20 THEN 'strong_trend_down'
                        WHEN close < ma20 THEN 'trend_down'
                        ELSE 'ranging'
                    END as regime
                FROM price_changes
            ),
            strategy_performance_by_regime AS (
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
                HAVING COUNT(*) >= 5
            )
            SELECT 
                regime,
                COUNT(DISTINCT strat) as strategies,
                SUM(trades) as total_trades,
                ROUND(AVG(avg_return), 4) as avg_return
            FROM strategy_performance_by_regime
            GROUP BY regime
            ORDER BY avg_return DESC
            """
            
            regime_summary = self.con.execute(regime_query).df()
            print("Performance by Price Regime:")
            print(regime_summary.to_string(index=False))
            
            # Find regime specialists
            specialist_query = f"""
            WITH price_changes AS (
                SELECT 
                    bar_index,
                    close,
                    AVG(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as ma20,
                    STDDEV(close) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as std20
                FROM read_parquet('{data_path}')
            ),
            regimes AS (
                SELECT 
                    bar_index,
                    CASE 
                        WHEN close > ma20 + std20 THEN 'strong_trend_up'
                        WHEN close > ma20 THEN 'trend_up'
                        WHEN close < ma20 - std20 THEN 'strong_trend_down'
                        WHEN close < ma20 THEN 'trend_down'
                        ELSE 'ranging'
                    END as regime
                FROM price_changes
            ),
            regime_specialists AS (
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
                HAVING COUNT(*) >= 10 AND AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) > 0.01  -- Profitable after Alpaca slippage
            )
            SELECT 
                regime,
                strat,
                trades,
                ROUND(avg_return, 4) as avg_return_pct
            FROM regime_specialists
            ORDER BY regime, avg_return DESC
            """
            
            specialists = self.con.execute(specialist_query).df()
            print("\n\nRegime Specialists (>0.01% return after slippage):")
            for regime in specialists['regime'].unique():
                regime_df = specialists[specialists['regime'] == regime]
                print(f"\n{regime}:")
                print(regime_df.head(5).to_string(index=False))
    
    def find_conditional_strategies(self, data_path: str):
        """Find strategies that work better under specific conditions."""
        
        print("\n\n=== Conditional Strategy Analysis ===\n")
        
        # Strategies that work when momentum is present
        query = f"""
        WITH momentum_context AS (
            -- Identify when momentum strategies are signaling
            SELECT 
                idx,
                MAX(val) as momentum_signal
            FROM read_parquet('{self.workspace_path}/traces/*/signals/momentum_grid/*.parquet')
            GROUP BY idx
        ),
        conditional_performance AS (
            SELECT 
                s.strat,
                CASE WHEN m.momentum_signal IS NOT NULL THEN 'with_momentum' ELSE 'no_momentum' END as context,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/rsi_grid/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            LEFT JOIN momentum_context m ON s.idx = m.idx
            WHERE s.val != 0
            GROUP BY s.strat, context
            HAVING COUNT(*) >= 20
        )
        SELECT 
            strat,
            MAX(CASE WHEN context = 'with_momentum' THEN avg_return END) as return_with_momentum,
            MAX(CASE WHEN context = 'no_momentum' THEN avg_return END) as return_no_momentum,
            MAX(CASE WHEN context = 'with_momentum' THEN avg_return END) - 
            MAX(CASE WHEN context = 'no_momentum' THEN avg_return END) as improvement
        FROM conditional_performance
        GROUP BY strat
        HAVING improvement > 0
        ORDER BY improvement DESC
        LIMIT 10
        """
        
        conditional = self.con.execute(query).df()
        print("RSI Strategies that improve with momentum context:")
        print(conditional.to_string(index=False))
    
    def close(self):
        self.con.close()


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python advanced_pattern_mining.py <workspace_path> <data_path>")
        sys.exit(1)
    
    miner = AdvancedPatternMiner(sys.argv[1])
    
    try:
        miner.analyze_composite_strategies(sys.argv[2])
        miner.analyze_regime_performance(sys.argv[2])
        miner.find_conditional_strategies(sys.argv[2])
    finally:
        miner.close()


if __name__ == "__main__":
    main()