import duckdb
import pandas as pd

def analyze_strategies(workspace_path: str, data_path: str):
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    # Get ALL strategies with basic stats
    query = f"""
    WITH strategy_stats AS (
        SELECT 
            s.strat,
            COUNT(*) as trades,
            MIN(s.idx) as first_trade,
            MAX(s.idx) as last_trade,
            AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as gross_return_pct
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
        JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
        WHERE s.val != 0
        GROUP BY s.strat
    )
    SELECT 
        strat,
        trades,
        ROUND(gross_return_pct, 5) as avg_return_per_trade,
        ROUND(gross_return_pct * trades, 3) as total_return_pct,
        ROUND(trades * 1.0 / (last_trade - first_trade) * 390, 1) as trades_per_day
    FROM strategy_stats
    ORDER BY total_return_pct DESC
    """
    
    df = con.execute(query).df()
    
    print("=== Strategy Performance Overview ===\n")
    
    # Group by strategy type
    df['strategy_type'] = df['strat'].apply(lambda x: 
        'rsi' if 'rsi' in x else
        'ma_crossover' if 'ma_crossover' in x else
        'momentum' if 'momentum' in x else
        'mean_reversion' if 'mean_reversion' in x else
        'breakout' if 'breakout' in x else 'other'
    )
    
    summary = df.groupby('strategy_type').agg({
        'trades': ['count', 'mean'],
        'avg_return_per_trade': ['mean', 'max'],
        'total_return_pct': ['mean', 'max'],
        'trades_per_day': 'mean'
    }).round(4)
    
    print("By Strategy Type:")
    print(summary)
    
    # Different filtering approaches
    print("\n\n=== Alternative Filtering Approaches ===")
    
    # 1. Low frequency, higher return strategies
    print("\n1. Low Frequency Strategies (< 1 trade/day, positive returns):")
    low_freq = df[(df['trades_per_day'] < 1) & (df['avg_return_per_trade'] > 0)]
    print(f"   Found {len(low_freq)} strategies")
    if len(low_freq) > 0:
        print(low_freq.nlargest(10, 'avg_return_per_trade')[['strat', 'trades', 'avg_return_per_trade', 'total_return_pct']].to_string(index=False))
    
    # 2. Total return approach (accumulation over time)
    print("\n2. Best Total Return Strategies:")
    best_total = df.nlargest(10, 'total_return_pct')
    print(best_total[['strat', 'trades', 'avg_return_per_trade', 'total_return_pct']].to_string(index=False))
    
    # 3. Break-even analysis
    print("\n3. Break-even Cost Analysis:")
    cost_levels = [0.1, 0.25, 0.5, 1.0, 2.0]  # in basis points
    for cost_bp in cost_levels:
        profitable = (df['avg_return_per_trade'] > cost_bp/100).sum()
        print(f"   Profitable with {cost_bp}bp round-trip cost: {profitable} strategies")
    
    # 4. Optimal trade frequency
    print("\n4. Trade Frequency vs Returns:")
    freq_bins = pd.cut(df['trades_per_day'], bins=[0, 0.5, 1, 2, 5, 100], labels=['<0.5/day', '0.5-1/day', '1-2/day', '2-5/day', '>5/day'])
    freq_analysis = df.groupby(freq_bins)['avg_return_per_trade'].agg(['count', 'mean', 'max']).round(5)
    print(freq_analysis)
    
    con.close()
    return df

if __name__ == "__main__":
    df = analyze_strategies('workspaces/expansive_grid_search_8d35f12c', 'data/SPY_1m.parquet')