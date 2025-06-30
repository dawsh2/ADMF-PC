"""Compare performance across multiple strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

# Known strategies to analyze
strategies = [
    {
        'name': 'Bollinger RSI Simple Signals',
        'workspace': 'signal_generation_7ecda4b8',
        'path': 'traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet'
    },
    {
        'name': 'RSI Bands',
        'workspace': 'signal_generation_4119862e',
        'path': 'traces/SPY_1m/signals/rsi_bands/SPY_compiled_strategy_0.parquet'
    }
]

# Add more strategies by searching
import os
import json

# Find unique single-strategy workspaces
found_strategies = {}
workspace_dir = Path('workspaces')

for workspace in workspace_dir.glob('signal_generation_*'):
    metadata_file = workspace / 'metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                
            # Only process if single strategy and has meaningful data
            if len(data.get('components', {})) == 1 and data.get('total_bars', 0) > 1000:
                comp_key = list(data['components'].keys())[0]
                comp = data['components'][comp_key]
                
                strat_type = comp.get('strategy_type', 'unknown')
                if strat_type not in ['unknown', 'bollinger_rsi_simple_signals', 'rsi_bands'] and strat_type not in found_strategies:
                    found_strategies[strat_type] = {
                        'name': strat_type.replace('_', ' ').title(),
                        'workspace': workspace.name.split('_')[-1],
                        'path': comp['signal_file_path']
                    }
        except:
            pass

# Add found strategies
for strat_type, strat_info in list(found_strategies.items())[:5]:  # Limit to 5 more
    strategies.append(strat_info)

print("=== Multi-Strategy Performance Comparison ===\n")

results = []

for strategy in strategies:
    try:
        # Load signal data
        signal_file = Path(f"workspaces/signal_generation_{strategy['workspace']}") / strategy['path']
        if not signal_file.exists():
            continue
            
        signals_df = pd.read_parquet(signal_file)
        signals_df['ts'] = pd.to_datetime(signals_df['ts'])
        
        # Convert to trades
        trades = []
        current_position = 0
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            new_signal = row['val']
            
            if current_position != 0 and new_signal != current_position:
                entry_idx = i - 1
                entry_row = signals_df.iloc[entry_idx]
                
                entry_price = entry_row['px']
                exit_price = row['px']
                pnl_pct = (exit_price / entry_price - 1) * current_position * 100
                bars_held = row['idx'] - entry_row['idx']
                
                trades.append({
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'direction': 'long' if current_position > 0 else 'short'
                })
            
            current_position = new_signal
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            # Calculate metrics
            total_trades = len(trades_df)
            avg_return = trades_df['pnl_pct'].mean()
            win_rate = (trades_df['pnl_pct'] > 0).mean()
            avg_bars = trades_df['bars_held'].mean()
            
            # Estimate annualized
            date_range = signals_df['ts'].max() - signals_df['ts'].min()
            days = date_range.days
            if days > 0:
                trades_per_year = total_trades / days * 365.25
                
                # Annual returns
                avg_decimal = avg_return / 100
                annual_no_cost = (1 + avg_decimal) ** trades_per_year - 1
                annual_1bp = (1 + avg_decimal - 0.0002) ** trades_per_year - 1
                
                # Sharpe estimate (rough)
                sharpe_estimate = np.sqrt(252) * avg_return / trades_df['pnl_pct'].std() if trades_df['pnl_pct'].std() > 0 else 0
                
                results.append({
                    'Strategy': strategy['name'],
                    'Trades': total_trades,
                    'Trades/Year': int(trades_per_year),
                    'Avg Return': avg_return,
                    'Win Rate': win_rate * 100,
                    'Avg Bars': avg_bars,
                    'Annual (0bp)': annual_no_cost * 100,
                    'Annual (1bp)': annual_1bp * 100,
                    'Sharpe': sharpe_estimate
                })
    except Exception as e:
        print(f"Error processing {strategy['name']}: {e}")

# Create comparison table
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('Annual (0bp)', ascending=False)
    
    print(f"{'Strategy':<35} {'Trades/Yr':<10} {'Avg Ret':<10} {'Win Rate':<10} {'Annual 0bp':<12} {'Annual 1bp':<12} {'Sharpe':<8}")
    print("-" * 110)
    
    for _, row in df.iterrows():
        print(f"{row['Strategy']:<35} {row['Trades/Year']:<10} {row['Avg Return']:>8.4f}%  {row['Win Rate']:>8.1f}%  "
              f"{row['Annual (0bp)']:>10.1f}%  {row['Annual (1bp)']:>10.1f}%  {row['Sharpe']:>6.2f}")
    
    print("\n=== KEY INSIGHTS ===")
    print(f"\nBest performer (no costs): {df.iloc[0]['Strategy']} at {df.iloc[0]['Annual (0bp)']:.1f}% annual")
    print(f"Best performer (1bp costs): {df.iloc[0]['Strategy']} at {df.iloc[0]['Annual (1bp)']:.1f}% annual")
    
    # Which strategies survive costs?
    survivors = df[df['Annual (1bp)'] > 0]
    print(f"\nStrategies profitable at 1bp cost: {len(survivors)} out of {len(df)}")
    
    if len(survivors) > 0:
        print("Profitable strategies:")
        for _, row in survivors.iterrows():
            print(f"  - {row['Strategy']}: {row['Annual (1bp)']:.1f}% annual")
    
    # Save results
    df.to_csv('strategy_comparison.csv', index=False)
    print(f"\nSaved comparison to 'strategy_comparison.csv'")
else:
    print("No valid strategies found to compare")