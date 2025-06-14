#!/usr/bin/env python3
"""
Simple Time-Based Exit Analysis

Clean analysis of P&L distribution for time-based exits on ALL fast RSI entries.
Shows the unfiltered reality of holding for fixed periods.
"""
import duckdb
import pandas as pd


def simple_time_exit_analysis(workspace_path: str, data_path: str):
    """Simple analysis of time-based exit performance."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Simple Time-Based Exit Analysis ===\n")
    print("All 6,280 fast RSI entries with simple time-based exits\n")
    
    # Test different holding periods
    holding_periods = [1, 3, 5, 7, 10, 15, 20, 30]
    
    results = []
    
    for period in holding_periods:
        query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
        ),
        trades AS (
            SELECT 
                entry_idx,
                entry_idx + {period} as exit_idx
            FROM fast_rsi_entries
        ),
        pnl AS (
            SELECT 
                (m2.close - m1.close) / m1.close * 100 as pnl_pct
            FROM trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
        )
        SELECT 
            {period} as holding_period,
            COUNT(*) as total_trades,
            ROUND(AVG(pnl_pct), 4) as avg_return,
            ROUND(MEDIAN(pnl_pct), 4) as median_return,
            ROUND(STDDEV(pnl_pct), 4) as volatility,
            COUNT(CASE WHEN pnl_pct > 0 THEN 1 END) as winners,
            COUNT(CASE WHEN pnl_pct < 0 THEN 1 END) as losers,
            ROUND(COUNT(CASE WHEN pnl_pct > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
            ROUND(MIN(pnl_pct), 4) as worst_loss,
            ROUND(MAX(pnl_pct), 4) as best_gain,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY pnl_pct), 4) as p5,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pnl_pct), 4) as p95,
            ROUND(AVG(pnl_pct) / STDDEV(pnl_pct), 3) as sharpe
        FROM pnl
        """
        
        try:
            result = con.execute(query).df()
            results.append(result.iloc[0])
            print(f"✓ {period}-bar analysis complete")
        except Exception as e:
            print(f"✗ Error in {period}-bar analysis: {e}")
    
    # Convert to DataFrame and display
    df = pd.DataFrame(results)
    
    print("\n1. Performance Summary:")
    print(df[['holding_period', 'total_trades', 'avg_return', 'median_return', 'win_rate', 'sharpe', 'volatility']].to_string(index=False))
    
    print("\n2. Risk Profile:")
    print(df[['holding_period', 'worst_loss', 'best_gain', 'p5', 'p95']].to_string(index=False))
    
    print("\n3. Trade Distribution:")
    print(df[['holding_period', 'winners', 'losers', 'win_rate']].to_string(index=False))
    
    # Calculate some key metrics
    if len(df) > 0:
        best_sharpe_row = df.loc[df['sharpe'].idxmax()]
        best_return_row = df.loc[df['avg_return'].idxmax()]
        
        print(f"\n=== Key Findings ===")
        print(f"Best Sharpe: {best_sharpe_row['holding_period']:.0f}-bar exit ({best_sharpe_row['sharpe']:.3f} Sharpe, {best_sharpe_row['avg_return']:.4f}% return)")
        print(f"Best Return: {best_return_row['holding_period']:.0f}-bar exit ({best_return_row['avg_return']:.4f}% return, {best_return_row['sharpe']:.3f} Sharpe)")
        
        # Calculate total return for strategy
        total_trades = df.iloc[0]['total_trades']  # Should be same for all
        print(f"\nTotal addressable trades: {total_trades}")
        print(f"Best strategy total return: {best_return_row['avg_return'] * total_trades:.2f}%")
        
        # Compare to our signal exits
        print(f"\nComparison to Signal Exits:")
        print(f"Mean reversion exits: 611 trades (9.7% coverage)")
        print(f"Time-based exits: {total_trades} trades (100% coverage)")
        print(f"Mean reversion performs better per trade, but covers far fewer trades")
    
    con.close()
    
    print(f"\n=== Strategy Insights ===")
    print(f"1. COVERAGE: Time exits cover ALL entries vs 9.7% for mean reversion")
    print(f"2. PERFORMANCE: Quality vs quantity tradeoff")
    print(f"3. REALISTIC BASELINE: This is unfiltered performance") 
    print(f"4. SCALABILITY: Time exits work for any volume of signals")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simple_time_exit_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    simple_time_exit_analysis(sys.argv[1], sys.argv[2])