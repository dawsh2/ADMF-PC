#!/usr/bin/env python3
"""
Simple MAE/MFE Analysis

Analyzes Maximum Adverse Excursion and Maximum Favorable Excursion
for fast RSI trades to optimize stop losses and profit targets.
"""
import duckdb
import pandas as pd


def simple_mae_mfe_analysis(workspace_path: str, data_path: str):
    """Simple MAE/MFE analysis to find optimal stop/target levels."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Simple MAE/MFE Analysis ===\n")
    
    # Analyze MAE/MFE for a sample of trades
    mae_mfe_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
        LIMIT 300  -- Sample for analysis
    ),
    trade_paths AS (
        SELECT 
            e.entry_idx,
            m.bar_index - e.entry_idx + 1 as bar_number,
            (m.close - m1.close) / m1.close * 100 as unrealized_pnl
        FROM fast_rsi_entries e
        JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
    ),
    mae_mfe_per_trade AS (
        SELECT 
            entry_idx,
            MIN(unrealized_pnl) as mae,  -- Maximum Adverse Excursion (worst drawdown)
            MAX(unrealized_pnl) as mfe,  -- Maximum Favorable Excursion (best profit)
            AVG(CASE WHEN bar_number = 18 THEN unrealized_pnl END) as pnl_18bar,
            AVG(CASE WHEN bar_number = 20 THEN unrealized_pnl END) as pnl_20bar
        FROM trade_paths
        GROUP BY entry_idx
    )
    SELECT 
        COUNT(*) as total_trades,
        ROUND(AVG(mae), 4) as avg_mae,
        ROUND(AVG(mfe), 4) as avg_mfe,
        ROUND(AVG(pnl_18bar), 4) as avg_18bar_pnl,
        ROUND(AVG(pnl_20bar), 4) as avg_20bar_pnl,
        ROUND(PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY mae), 4) as mae_p10,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mae), 4) as mae_p25,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mae), 4) as mae_p50,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mfe), 4) as mfe_p75,
        ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY mfe), 4) as mfe_p90,
        ROUND(MIN(mae), 4) as worst_mae,
        ROUND(MAX(mfe), 4) as best_mfe
    FROM mae_mfe_per_trade
    """
    
    mae_mfe_summary = con.execute(mae_mfe_query).df()
    print("MAE/MFE Summary (300 trade sample):")
    print(mae_mfe_summary.to_string(index=False))
    
    # Analyze stop loss effectiveness
    print("\n\nStop Loss Analysis:")
    
    stop_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    for stop_pct in stop_levels:
        stop_query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
            LIMIT 300
        ),
        trade_paths AS (
            SELECT 
                e.entry_idx,
                m.bar_index - e.entry_idx + 1 as bar_number,
                (m.close - m1.close) / m1.close * 100 as unrealized_pnl
            FROM fast_rsi_entries e
            JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
        ),
        stop_analysis AS (
            SELECT 
                entry_idx,
                MIN(CASE WHEN unrealized_pnl <= -{stop_pct} THEN bar_number END) as stop_hit_bar,
                MIN(unrealized_pnl) as mae,
                AVG(CASE WHEN bar_number = 18 THEN unrealized_pnl END) as final_pnl
            FROM trade_paths
            GROUP BY entry_idx
        )
        SELECT 
            COUNT(*) as total_trades,
            COUNT(stop_hit_bar) as stop_hits,
            ROUND(COUNT(stop_hit_bar) * 100.0 / COUNT(*), 2) as stop_hit_rate,
            ROUND(AVG(CASE WHEN stop_hit_bar IS NOT NULL THEN -{stop_pct} ELSE final_pnl END), 4) as avg_pnl_with_stop,
            ROUND(AVG(final_pnl), 4) as avg_pnl_no_stop,
            ROUND(AVG(CASE WHEN stop_hit_bar IS NOT NULL THEN stop_hit_bar END), 1) as avg_stop_timing
        FROM stop_analysis
        """
        
        try:
            stop_result = con.execute(stop_query).df()
            print(f"\n{stop_pct}% Stop Loss:")
            print(stop_result.to_string(index=False))
        except Exception as e:
            print(f"Error analyzing {stop_pct}% stop: {e}")
    
    # Analyze profit target effectiveness  
    print("\n\nProfit Target Analysis:")
    
    target_levels = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    for target_pct in target_levels:
        target_query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
            LIMIT 300
        ),
        trade_paths AS (
            SELECT 
                e.entry_idx,
                m.bar_index - e.entry_idx + 1 as bar_number,
                (m.close - m1.close) / m1.close * 100 as unrealized_pnl
            FROM fast_rsi_entries e
            JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
        ),
        target_analysis AS (
            SELECT 
                entry_idx,
                MIN(CASE WHEN unrealized_pnl >= {target_pct} THEN bar_number END) as target_hit_bar,
                MAX(unrealized_pnl) as mfe,
                AVG(CASE WHEN bar_number = 18 THEN unrealized_pnl END) as final_pnl
            FROM trade_paths
            GROUP BY entry_idx
        )
        SELECT 
            COUNT(*) as total_trades,
            COUNT(target_hit_bar) as target_hits,
            ROUND(COUNT(target_hit_bar) * 100.0 / COUNT(*), 2) as target_hit_rate,
            ROUND(AVG(CASE WHEN target_hit_bar IS NOT NULL THEN {target_pct} ELSE final_pnl END), 4) as avg_pnl_with_target,
            ROUND(AVG(final_pnl), 4) as avg_pnl_no_target,
            ROUND(AVG(CASE WHEN target_hit_bar IS NOT NULL THEN target_hit_bar END), 1) as avg_target_timing
        FROM target_analysis
        """
        
        try:
            target_result = con.execute(target_query).df()
            print(f"\n{target_pct}% Profit Target:")
            print(target_result.to_string(index=False))
        except Exception as e:
            print(f"Error analyzing {target_pct}% target: {e}")
    
    con.close()
    
    print(f"\n=== Optimization Insights ===")
    print(f"1. MAE ANALYSIS: Shows typical adverse moves to set stop losses")
    print(f"2. MFE ANALYSIS: Shows profit potential to set targets") 
    print(f"3. STOP OPTIMIZATION: Balance protection vs whipsaws")
    print(f"4. TARGET OPTIMIZATION: Balance early profits vs letting winners run")
    print(f"5. COMBINED APPROACH: Use both stops and targets for complete framework")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simple_mae_mfe_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    simple_mae_mfe_analysis(sys.argv[1], sys.argv[2])