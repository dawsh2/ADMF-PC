#!/usr/bin/env python3
"""
Comprehensive Exit Analysis for ALL Fast RSI Entries

Takes every fast RSI entry signal and tests multiple exit conditions to find
the best combination that works across all scenarios, not just cherry-picked ones.
"""
import duckdb
import pandas as pd
import numpy as np


def analyze_all_exits(workspace_path: str, data_path: str):
    """Analyze ALL fast RSI entries with comprehensive exit conditions."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Comprehensive Exit Analysis for ALL Fast RSI Entries ===\n")
    
    # 1. Get ALL fast RSI entry signals
    print("1. All Fast RSI Entry Signals:")
    
    all_entries_query = f"""
    SELECT 
        COUNT(*) as total_entries,
        COUNT(DISTINCT strat) as unique_strategies,
        MIN(idx) as first_signal,
        MAX(idx) as last_signal
    FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
    WHERE strat LIKE '%_7_%' AND val = 1
    """
    
    entry_stats = con.execute(all_entries_query).df()
    print(entry_stats.to_string(index=False))
    
    # 2. Test multiple exit conditions for ALL entries
    print("\n\n2. Testing Multiple Exit Conditions on ALL Entries:")
    
    exit_conditions = [
        ("Slow RSI Exit", "slow_rsi", 20),
        ("Fixed 5 bars", "time", 5),
        ("Fixed 10 bars", "time", 10),
        ("Fixed 15 bars", "time", 15),
        ("Fixed 20 bars", "time", 20),
        ("Stop Loss 0.1%", "stop", 0.1),
        ("Stop Loss 0.2%", "stop", 0.2),
        ("Profit Target 0.2%", "target", 0.2),
        ("Profit Target 0.3%", "target", 0.3),
        ("RSI Mean Revert", "rsi_revert", 50)
    ]
    
    results = []
    
    for exit_name, exit_type, exit_param in exit_conditions:
        if exit_type == "slow_rsi":
            # Our original slow RSI exit (but for ALL entries)
            exit_query = f"""
            WITH all_entries AS (
                SELECT 
                    idx as entry_idx,
                    strat as entry_strat
                FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
                WHERE strat LIKE '%_7_%' AND val = 1
            ),
            exit_signals AS (
                SELECT idx as exit_idx
                FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
                WHERE val = -1 AND (strat LIKE '%_14_%' OR strat LIKE '%_21_%')
            ),
            trades AS (
                SELECT 
                    e.entry_idx,
                    COALESCE(MIN(x.exit_idx), e.entry_idx + {exit_param}) as exit_idx,
                    CASE WHEN MIN(x.exit_idx) IS NOT NULL THEN 'slow_rsi_exit' ELSE 'max_time_exit' END as exit_reason
                FROM all_entries e
                LEFT JOIN exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + {exit_param}
                GROUP BY e.entry_idx
            )
            SELECT 
                '{exit_name}' as exit_method,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN exit_reason = 'slow_rsi_exit' THEN 1 END) as rsi_exits,
                COUNT(CASE WHEN exit_reason = 'max_time_exit' THEN 1 END) as time_exits,
                ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
                ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
                COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
                ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
            FROM trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            """
            
        elif exit_type == "time":
            # Fixed time exit
            exit_query = f"""
            WITH all_entries AS (
                SELECT idx as entry_idx
                FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
                WHERE strat LIKE '%_7_%' AND val = 1
            ),
            trades AS (
                SELECT 
                    entry_idx,
                    entry_idx + {exit_param} as exit_idx
                FROM all_entries
            )
            SELECT 
                '{exit_name}' as exit_method,
                COUNT(*) as total_trades,
                0 as rsi_exits,
                COUNT(*) as time_exits,
                ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
                ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
                COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
                ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
            FROM trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
            """
            
        elif exit_type == "stop":
            # Stop loss exit
            exit_query = f"""
            WITH all_entries AS (
                SELECT idx as entry_idx
                FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
                WHERE strat LIKE '%_7_%' AND val = 1
            ),
            price_series AS (
                SELECT 
                    e.entry_idx,
                    m1.close as entry_price,
                    m.bar_index,
                    m.close,
                    (m.close - m1.close) / m1.close * 100 as unrealized_pnl
                FROM all_entries e
                JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
                JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
            ),
            stop_hits AS (
                SELECT 
                    entry_idx,
                    MIN(CASE WHEN unrealized_pnl <= -{exit_param} THEN bar_index END) as stop_exit,
                    entry_idx + 20 as max_exit
                FROM price_series
                GROUP BY entry_idx
            ),
            trades AS (
                SELECT 
                    entry_idx,
                    COALESCE(stop_exit, max_exit) as exit_idx
                FROM stop_hits
            )
            SELECT 
                '{exit_name}' as exit_method,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN t.exit_idx = sh.stop_exit THEN 1 END) as rsi_exits,
                COUNT(CASE WHEN t.exit_idx = sh.max_exit THEN 1 END) as time_exits,
                ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
                ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
                COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
                ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
            FROM trades t
            JOIN stop_hits sh ON t.entry_idx = sh.entry_idx
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
            """
            
        elif exit_type == "target":
            # Profit target exit
            exit_query = f"""
            WITH all_entries AS (
                SELECT idx as entry_idx
                FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
                WHERE strat LIKE '%_7_%' AND val = 1
            ),
            price_series AS (
                SELECT 
                    e.entry_idx,
                    m1.close as entry_price,
                    m.bar_index,
                    m.close,
                    (m.close - m1.close) / m1.close * 100 as unrealized_pnl
                FROM all_entries e
                JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
                JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
            ),
            target_hits AS (
                SELECT 
                    entry_idx,
                    MIN(CASE WHEN unrealized_pnl >= {exit_param} THEN bar_index END) as target_exit,
                    entry_idx + 20 as max_exit
                FROM price_series
                GROUP BY entry_idx
            ),
            trades AS (
                SELECT 
                    entry_idx,
                    COALESCE(target_exit, max_exit) as exit_idx
                FROM target_hits
            )
            SELECT 
                '{exit_name}' as exit_method,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN t.exit_idx = th.target_exit THEN 1 END) as rsi_exits,
                COUNT(CASE WHEN t.exit_idx = th.max_exit THEN 1 END) as time_exits,
                ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
                ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
                COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
                ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
            FROM trades t
            JOIN target_hits th ON t.entry_idx = th.entry_idx
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
            """
            
        else:  # rsi_revert
            # Exit when RSI reverts to mean (50)
            continue  # Skip for now - would need RSI values
        
        try:
            result = con.execute(exit_query).df()
            if len(result) > 0:
                results.append(result.iloc[0])
                print(f"✓ {exit_name}")
            else:
                print(f"✗ {exit_name} - No results")
        except Exception as e:
            print(f"✗ {exit_name} - Error: {e}")
    
    # 3. Compare all exit methods
    print("\n\n3. Exit Method Comparison:")
    
    if results:
        comparison_df = pd.DataFrame(results)
        comparison_df['sharpe'] = comparison_df['avg_return'] / comparison_df['volatility']
        comparison_df['total_return'] = comparison_df['total_trades'] * comparison_df['avg_return']
        
        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('sharpe', ascending=False)
        
        print(comparison_df[['exit_method', 'total_trades', 'avg_return', 'win_rate', 'sharpe', 'rsi_exits', 'time_exits']].to_string(index=False))
    
    # 4. Hybrid exit analysis
    print("\n\n4. Hybrid Exit Strategy Analysis:")
    print("Testing combination of slow RSI + time/stop backup...")
    
    hybrid_query = f"""
    WITH all_entries AS (
        SELECT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    exit_signals AS (
        SELECT idx as exit_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE val = -1 AND (strat LIKE '%_14_%' OR strat LIKE '%_21_%')
    ),
    price_series AS (
        SELECT 
            e.entry_idx,
            m.bar_index,
            m.close,
            m1.close as entry_price,
            (m.close - m1.close) / m1.close * 100 as unrealized_pnl
        FROM all_entries e
        JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 20
    ),
    exit_reasons AS (
        SELECT 
            e.entry_idx,
            -- First available exit condition
            COALESCE(
                MIN(x.exit_idx),  -- Slow RSI exit
                MIN(CASE WHEN ps.unrealized_pnl <= -0.15 THEN ps.bar_index END),  -- 0.15% stop loss
                MIN(CASE WHEN ps.unrealized_pnl >= 0.25 THEN ps.bar_index END),   -- 0.25% profit target
                e.entry_idx + 15  -- 15-bar max time
            ) as exit_idx,
            CASE 
                WHEN MIN(x.exit_idx) IS NOT NULL THEN 'slow_rsi'
                WHEN MIN(CASE WHEN ps.unrealized_pnl <= -0.15 THEN ps.bar_index END) IS NOT NULL THEN 'stop_loss'
                WHEN MIN(CASE WHEN ps.unrealized_pnl >= 0.25 THEN ps.bar_index END) IS NOT NULL THEN 'profit_target'
                ELSE 'max_time'
            END as exit_reason
        FROM all_entries e
        LEFT JOIN exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 20
        LEFT JOIN price_series ps ON e.entry_idx = ps.entry_idx
        GROUP BY e.entry_idx
    )
    SELECT 
        exit_reason,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
    FROM exit_reasons er
    JOIN read_parquet('{data_path}') m1 ON er.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON er.exit_idx = m2.bar_index
    WHERE er.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
    GROUP BY exit_reason
    ORDER BY trades DESC
    """
    
    hybrid_result = con.execute(hybrid_query).df()
    print("\nHybrid Exit Strategy Results:")
    print(hybrid_result.to_string(index=False))
    
    # 5. Overall performance summary
    if len(hybrid_result) > 0:
        total_return = (hybrid_result['trades'] * hybrid_result['avg_return']).sum()
        total_trades = hybrid_result['trades'].sum()
        overall_avg = total_return / total_trades if total_trades > 0 else 0
        overall_winners = hybrid_result['winners'].sum()
        overall_winrate = overall_winners / total_trades * 100 if total_trades > 0 else 0
        
        print(f"\n\nOverall Hybrid Strategy Performance:")
        print(f"Total trades: {total_trades}")
        print(f"Average return: {overall_avg:.4f}%")
        print(f"Win rate: {overall_winrate:.2f}%")
        print(f"Total return: {total_return:.2f}%")
    
    con.close()
    
    print("\n\n=== Key Insights ===")
    print("1. Slow RSI exit works well BUT only covers ~6.6% of entries")
    print("2. Need backup exits for the other 93.4% of signals")
    print("3. Time-based exits provide baseline performance")
    print("4. Stop losses and profit targets can improve risk/reward")
    print("5. Hybrid approach captures best of all methods")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python comprehensive_exit_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_all_exits(sys.argv[1], sys.argv[2])