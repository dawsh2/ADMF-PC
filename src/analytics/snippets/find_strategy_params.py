# Find Strategy Parameters by Hash
# Quick tool to look up parameters for specific strategy hashes

import pandas as pd

def find_strategy_details(strategy_hash_partial, performance_df=None, strategy_index=None):
    """
    Find strategy details by partial hash match
    
    Args:
        strategy_hash_partial: Partial hash to search for (e.g., '3bee7e1f')
        performance_df: Performance dataframe (uses global if not provided)
        strategy_index: Strategy index (uses global if not provided)
    """
    # Use global variables if not provided
    if performance_df is None:
        performance_df = globals().get('performance_df', pd.DataFrame())
    if strategy_index is None:
        strategy_index = globals().get('strategy_index', pd.DataFrame())
    
    # Search in performance_df first
    if len(performance_df) > 0:
        matches = performance_df[performance_df['strategy_hash'].str.contains(strategy_hash_partial, na=False)]
        
        if len(matches) > 0:
            print(f"✅ Found {len(matches)} matches in performance data:")
            for idx, row in matches.iterrows():
                print(f"\n{'='*60}")
                print(f"Strategy: {row['strategy_type']} - {row['strategy_hash']}")
                print(f"{'='*60}")
                
                # Performance metrics
                print("\n📊 Performance Metrics:")
                print(f"  Sharpe Ratio: {row.get('sharpe_ratio', 'N/A'):.2f}")
                print(f"  Total Return: {row.get('total_return', 0)*100:.2f}%")
                print(f"  Win Rate: {row.get('win_rate', 0)*100:.1f}%")
                print(f"  Profit Factor: {row.get('profit_factor', 0):.2f}")
                print(f"  Number of Trades: {row.get('num_trades', 0)}")
                if 'num_trades' in row:
                    trades_per_day = row['num_trades'] / (len(market_data) / 78) if 'market_data' in globals() else 'N/A'
                    print(f"  Trades per Day: {trades_per_day:.1f}" if isinstance(trades_per_day, (int, float)) else "  Trades per Day: N/A")
                
                # Parameters
                print("\n🔧 Parameters:")
                param_found = False
                
                # Check common parameter names
                param_names = ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier', 
                              'lookback', 'threshold', 'oversold', 'overbought',
                              'param_period', 'param_std_dev', 'param_fast_period', 'param_slow_period']
                
                for param in param_names:
                    if param in row and pd.notna(row[param]):
                        print(f"  {param}: {row[param]}")
                        param_found = True
                
                if not param_found:
                    # Try to find any column that might be a parameter
                    for col in row.index:
                        if ('param' in col.lower() or col in ['period', 'std_dev', 'multiplier']) and pd.notna(row[col]):
                            print(f"  {col}: {row[col]}")
                
                # Additional info
                if 'total_execution_cost' in row:
                    print(f"\n💰 Execution Cost: {row['total_execution_cost']*100:.3f}%")
                
            return matches
    
    # If not found in performance_df, check strategy_index
    if len(strategy_index) > 0:
        matches = strategy_index[strategy_index['strategy_hash'].str.contains(strategy_hash_partial, na=False)]
        
        if len(matches) > 0:
            print(f"\n✅ Found {len(matches)} matches in strategy index:")
            for idx, row in matches.iterrows():
                print(f"\nStrategy: {row['strategy_type']} - {row['strategy_hash']}")
                print("Parameters from index:")
                
                # Show all non-null columns that might be parameters
                for col in row.index:
                    if col not in ['strategy_hash', 'strategy_type', 'trace_path'] and pd.notna(row[col]):
                        print(f"  {col}: {row[col]}")
            
            return matches
    
    print(f"❌ No strategy found matching '{strategy_hash_partial}'")
    return pd.DataFrame()

# Automatically search for the specific strategy mentioned
target_hash = '3bee7e1f'
print(f"🔍 Looking up strategy {target_hash}...")
print("=" * 80)

result = find_strategy_details(target_hash)

# Also show surrounding strategies by performance
if len(result) > 0 and len(performance_df) > 0:
    target_sharpe = result.iloc[0]['sharpe_ratio']
    print(f"\n📈 Strategies with similar performance:")
    print("=" * 60)
    
    # Find strategies within 0.1 Sharpe ratio
    similar = performance_df[
        (performance_df['sharpe_ratio'] >= target_sharpe - 0.1) & 
        (performance_df['sharpe_ratio'] <= target_sharpe + 0.1) &
        (performance_df['num_trades'] >= 1000)  # High frequency only
    ].sort_values('sharpe_ratio', ascending=False)
    
    for idx, row in similar.head(5).iterrows():
        if row['strategy_hash'] != result.iloc[0]['strategy_hash']:
            print(f"\n{row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Sharpe: {row['sharpe_ratio']:.2f}, Trades: {row['num_trades']}")
            for param in ['period', 'std_dev']:
                if param in row and pd.notna(row[param]):
                    print(f"  {param}: {row[param]}")