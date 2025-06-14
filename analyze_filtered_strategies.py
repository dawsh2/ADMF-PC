#!/usr/bin/env python3
"""
Analyze the filtered strategies to understand the filtering process.
"""
import duckdb
import pandas as pd

def analyze_filtering_stages(workspace_path: str, data_path: str, min_trades: int = 250):
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print(f"=== Analyzing Filter Stages (min_trades={min_trades}) ===\n")
    
    # Stage 1: Trade count filter
    trade_count_query = f"""
    SELECT 
        strat,
        COUNT(*) as trades
    FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
    WHERE val != 0
    GROUP BY strat
    HAVING COUNT(*) >= {min_trades}
    ORDER BY trades DESC
    """
    
    trade_filtered = con.execute(trade_count_query).df()
    print(f"1. Trade Count Filter: {len(trade_filtered)} strategies with >= {min_trades} trades")
    print(f"   Trade range: {trade_filtered['trades'].min()} - {trade_filtered['trades'].max()}")
    
    # Stage 2: Profitability filter (after costs)
    profit_query = f"""
    WITH performance AS (
        SELECT 
            s.strat,
            COUNT(*) as trades,
            AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as gross_return,
            AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100 - 0.02
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100 - 0.02
            END) as net_return,
            STDDEV(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as volatility
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
        JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
        WHERE s.val != 0
        GROUP BY s.strat
        HAVING COUNT(*) >= {min_trades}
    )
    SELECT 
        strat,
        trades,
        ROUND(gross_return, 4) as gross_return_pct,
        ROUND(net_return, 4) as net_return_pct,
        ROUND(volatility, 4) as volatility_pct,
        ROUND(net_return / NULLIF(volatility, 0), 3) as sharpe_ratio
    FROM performance
    WHERE net_return > 0
    ORDER BY net_return DESC
    """
    
    profitable = con.execute(profit_query).df()
    print(f"\n2. Profitability Filter: {len(profitable)} strategies profitable after 2bp costs")
    
    if len(profitable) > 0:
        print("\nTop 10 Profitable Strategies:")
        print(profitable.head(10).to_string(index=False))
        
        # Show distribution
        print(f"\nReturn distribution:")
        print(f"   Best net return: {profitable['net_return_pct'].max():.4f}%")
        print(f"   Median net return: {profitable['net_return_pct'].median():.4f}%")
        print(f"   Worst net return: {profitable['net_return_pct'].min():.4f}%")
        print(f"   Best Sharpe: {profitable['sharpe_ratio'].max():.3f}")
    
    # Stage 3: Show correlation analysis for top strategies
    if len(profitable) >= 4:
        print("\n3. Correlation Analysis of Top Strategies:")
        
        # Get signal overlap for top strategies
        top_strategies = profitable.head(10)['strat'].tolist()
        strategy_list = "'" + "','".join(top_strategies) + "'"
        
        overlap_query = f"""
        WITH strategy_signals AS (
            SELECT 
                strat,
                ARRAY_AGG(idx) as indices,
                COUNT(*) as count
            FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
            WHERE val != 0 AND strat IN ({strategy_list})
            GROUP BY strat
        ),
        overlap_matrix AS (
            SELECT 
                s1.strat as strat1,
                s2.strat as strat2,
                s1.count as count1,
                s2.count as count2,
                LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) as overlap_count,
                ROUND(LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) * 100.0 / 
                    LEAST(s1.count, s2.count), 1) as overlap_pct
            FROM strategy_signals s1
            CROSS JOIN strategy_signals s2
            WHERE s1.strat < s2.strat
        )
        SELECT * FROM overlap_matrix
        WHERE overlap_pct > 20
        ORDER BY overlap_pct DESC
        LIMIT 20
        """
        
        overlaps = con.execute(overlap_query).df()
        
        if len(overlaps) > 0:
            print("\nHigh correlations (>20% signal overlap):")
            print(overlaps.to_string(index=False))
        else:
            print("\nNo high correlations found among top strategies!")
    
    con.close()
    
    # Return the profitable strategies for further analysis
    return profitable


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_filtered_strategies.py <workspace_path> <data_path> [min_trades]")
        sys.exit(1)
    
    workspace = sys.argv[1]
    data = sys.argv[2]
    min_trades = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    
    analyze_filtering_stages(workspace, data, min_trades)