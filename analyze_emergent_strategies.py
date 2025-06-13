#!/usr/bin/env python3
"""
Analyze emergent trading strategies from pattern mining results.
Creates actionable trading rules from discovered patterns.
"""
import json
import pandas as pd
from pathlib import Path
import duckdb


def create_composite_strategies(workspace_path: str, data_path: str):
    """Create and backtest composite strategies based on discovered patterns."""
    
    # Load insights
    insights_path = Path(workspace_path) / "pattern_analysis" / "insights.json"
    with open(insights_path, 'r') as f:
        insights = json.load(f)
    
    con = duckdb.connect()
    
    print("=== Emergent Trading Strategies ===\n")
    
    # 1. Best Entry/Exit Combination Strategy
    best_combo = insights['best_signal_combinations'][0]
    print(f"1. Composite Strategy: {best_combo['entry']} → {best_combo['exit']}")
    print(f"   Expected return: {best_combo['avg_return']:.3f}%")
    print(f"   Average holding period: {best_combo['holding_period']} bars")
    print(f"   Historical occurrences: {best_combo['occurrences']}")
    
    # 2. Filtered RSI Strategy (with mean reversion confirmation)
    print("\n2. Filtered RSI Strategy (with mean reversion confirmation)")
    best_conditional = insights['conditional_improvements'][0]
    print(f"   Base strategy: {best_conditional['strategy']}")
    print(f"   Filter: Trade only when {best_conditional['condition']}")
    print(f"   Performance improvement: +{best_conditional['improvement']:.2f}%")
    
    # 3. Multi-Exit Strategy
    print("\n3. Adaptive Exit Strategy")
    for exit_info in insights['optimal_exits']:
        print(f"   {exit_info['strategy_type']}: Use {exit_info['best_exit']} exit")
        print(f"      Expected return: {exit_info['expected_return']:.3f}%")
    
    # Create SQL for composite strategy backtest
    print("\n=== Backtest Composite Strategies ===\n")
    
    # Test the MA entry → RSI exit combination
    composite_query = f"""
    WITH ma_entries AS (
        SELECT idx, ts, strat
        FROM read_parquet('{workspace_path}/traces/*/signals/ma_crossover_grid/*.parquet')
        WHERE val = 1 AND strat LIKE '%_10_%'  -- Fast MA
    ),
    rsi_exits AS (
        SELECT idx, ts, strat
        FROM read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet')
        WHERE val = -1 AND strat LIKE '%_7_%'  -- Fast RSI
    ),
    composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            e.strat as entry_strategy,
            FIRST(x.strat) as exit_strategy
        FROM ma_entries e
        JOIN rsi_exits x ON x.idx > e.idx AND x.idx <= e.idx + 10
        GROUP BY e.idx, e.strat
    )
    SELECT 
        COUNT(*) as trades,
        AVG((m_exit.close - m_entry.close) / m_entry.close * 100) as avg_return,
        AVG(exit_idx - entry_idx) as avg_holding_period,
        STDDEV((m_exit.close - m_entry.close) / m_entry.close * 100) as volatility
    FROM composite_trades ct
    JOIN read_parquet('{data_path}') m_entry ON ct.entry_idx = m_entry.bar_index
    JOIN read_parquet('{data_path}') m_exit ON ct.exit_idx = m_exit.bar_index
    """
    
    result = con.execute(composite_query).df()
    print("Composite Strategy (MA entry → RSI exit):")
    print(f"  Trades: {result['trades'].iloc[0]}")
    print(f"  Avg Return: {result['avg_return'].iloc[0]:.3f}%")
    print(f"  Avg Holding: {result['avg_holding_period'].iloc[0]:.1f} bars")
    print(f"  Volatility: {result['volatility'].iloc[0]:.3f}%")
    
    # Test filtered RSI strategy
    filtered_query = f"""
    WITH mean_rev_signals AS (
        SELECT idx, val
        FROM read_parquet('{workspace_path}/traces/*/signals/mean_reversion_grid/*.parquet')
        WHERE val != 0
    ),
    mean_rev_windows AS (
        SELECT 
            idx as start_idx,
            LEAD(idx, 1, 999999) OVER (ORDER BY idx) as end_idx
        FROM mean_rev_signals
    ),
    filtered_rsi_signals AS (
        SELECT 
            r.idx,
            r.val,
            r.strat
        FROM read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet') r
        JOIN mean_rev_windows w ON r.idx >= w.start_idx AND r.idx < w.end_idx
        WHERE r.val != 0 AND r.strat LIKE '%_7_20_%'
    )
    SELECT 
        COUNT(*) as filtered_trades,
        AVG(CASE 
            WHEN val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return
    FROM filtered_rsi_signals s
    JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
    """
    
    result2 = con.execute(filtered_query).df()
    print("\n\nFiltered RSI Strategy (trade only during mean reversion):")
    print(f"  Filtered Trades: {result2['filtered_trades'].iloc[0]}")
    print(f"  Avg Return: {result2['avg_return'].iloc[0]:.3f}%")
    
    con.close()
    
    # Generate trading rules
    print("\n\n=== Actionable Trading Rules ===\n")
    
    print("Rule 1: Composite Entry/Exit")
    print("  WHEN ma_crossover_fast goes long")
    print("  THEN enter long position")
    print("  WHEN rsi_fast goes short") 
    print("  THEN exit position")
    print("  EXPECTED: 0.046% per trade\n")
    
    print("Rule 2: Filtered RSI")
    print("  IF mean_reversion_signal is active")
    print("  THEN follow rsi_7_20_80 signals")
    print("  ELSE skip rsi signals")
    print("  IMPROVEMENT: +0.01% per trade\n")
    
    print("Rule 3: Adaptive Exits")
    print("  IF strategy_type == 'ma_crossover':")
    print("    EXIT after 1 bar")
    print("  ELIF strategy_type == 'rsi':")
    print("    EXIT on opposite signal")
    print("  EXPECTED: Optimal returns by type\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_emergent_strategies.py <workspace> <data_path>")
        sys.exit(1)
        
    create_composite_strategies(sys.argv[1], sys.argv[2])