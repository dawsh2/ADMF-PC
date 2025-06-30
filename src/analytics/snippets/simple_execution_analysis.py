# Simple execution trace analysis - add this to trade_analysis.ipynb
# This version uses a simpler approach to extract trades from fills

import pandas as pd
import numpy as np
from pathlib import Path

# Load execution fills
results_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/latest')
fills_path = results_dir / 'traces/execution/fills'

print("Loading fills...")
fills_files = list(fills_path.glob('*.parquet'))
if fills_files:
    fills_list = []
    for f in fills_files:
        df = pd.read_parquet(f)
        fills_list.append(df)
    fills_df = pd.concat(fills_list, ignore_index=True)
    fills_df['timestamp'] = pd.to_datetime(fills_df['ts'])
    fills_df = fills_df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(fills_df)} fills")
    print(f"Columns: {fills_df.columns.tolist()}")
    print("\nFirst 5 fills:")
    print(fills_df.head())
else:
    print("❌ No fills found")
    fills_df = pd.DataFrame()

# Extract trades from fills
trades = []
current_position = None

for idx, fill in fills_df.iterrows():
    if len(trades) >= 100:
        break
    
    # Extract data from nested metadata
    meta = fill.get('metadata', {})
    if not isinstance(meta, dict):
        continue
        
    price = float(meta.get('price', 0))
    side = meta.get('side', '').lower()
    quantity = float(meta.get('quantity', 0))
    
    # Skip if no price
    if price == 0:
        continue
        
    # Get nested metadata for exit type
    nested_meta = meta.get('metadata', {})
    exit_type = nested_meta.get('exit_type', None)
    
    # Simple logic: alternating fills are entries and exits
    if current_position is None:
        # This is an entry
        current_position = {
            'entry_time': fill['timestamp'],
            'entry_price': price,
            'side': side,
            'quantity': quantity,
            'trade_num': len(trades) + 1
        }
    else:
        # This is an exit
        entry_price = current_position['entry_price']
        exit_price = price
        
        # Skip if prices are invalid
        if entry_price == 0 or exit_price == 0:
            print(f"Skipping trade with zero price: entry={entry_price}, exit={exit_price}")
            current_position = None
            continue
        
        # Determine direction
        if current_position['side'] in ['buy']:
            direction = 'LONG'
            raw_return = (exit_price - entry_price) / entry_price
        else:
            direction = 'SHORT'
            raw_return = (entry_price - exit_price) / entry_price
        
        # Apply execution cost
        net_return = raw_return - 0.0001
        
        # Determine exit type
        if exit_type is None:
            exit_type = 'signal'
            
        # Map exit types
        exit_type_map = {
            'stop_loss': 'stop',
            'take_profit': 'target',
            'signal': 'signal'
        }
        exit_type = exit_type_map.get(exit_type, exit_type)
        
        trade = {
            'num': current_position['trade_num'],
            'entry_time': current_position['entry_time'],
            'entry_price': entry_price,
            'exit_time': fill['timestamp'],
            'exit_price': exit_price,
            'dir': direction,
            'exit_type': exit_type,
            'return': net_return,
            'bars': max(1, int((fill['timestamp'] - current_position['entry_time']).total_seconds() / 300))
        }
        
        # Calculate stop and target prices
        if direction == 'LONG':
            trade['stop'] = entry_price * 0.99925
            trade['target'] = entry_price * 1.001
        else:
            trade['stop'] = entry_price * 1.00075
            trade['target'] = entry_price * 0.999
            
        trades.append(trade)
        current_position = None

# Convert to DataFrame
execution_trades = pd.DataFrame(trades)

if len(execution_trades) > 0:
    print(f"\n{'='*120}")
    print("EXECUTION ENGINE TRADE-BY-TRADE ANALYSIS")
    print(f"{'='*120}")
    
    print(f"\nFound {len(execution_trades)} trades")
    
    # Display first 30 trades
    print(f"\nFirst 30 trades from EXECUTION ENGINE:")
    print("-"*120)
    print(f"{'#':>3} {'Entry Time':>20} {'Dir':>5} {'Entry':>8} {'Stop':>8} {'Target':>8} "
          f"{'Exit':>6} {'Exit$':>8} {'Ret%':>8} {'Bars':>5}")
    print("-"*120)
    
    for _, t in execution_trades.head(30).iterrows():
        print(f"{t['num']:>3} {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>20} "
              f"{t['dir']:>5} {t['entry_price']:>8.2f} {t['stop']:>8.2f} {t['target']:>8.2f} "
              f"{t['exit_type']:>6} {t['exit_price']:>8.2f} {t['return']*100:>7.3f}% {t['bars']:>5}")
    
    # Summary statistics
    exit_counts = execution_trades['exit_type'].value_counts()
    print(f"\nExit Type Summary:")
    for exit_type in ['stop', 'target', 'signal']:
        count = exit_counts.get(exit_type, 0)
        print(f"  {exit_type:>8}: {count:>4} ({count/len(execution_trades)*100:>5.1f}%)")
    
    print(f"\nPerformance from EXECUTION ENGINE:")
    print(f"  Avg return per trade: {execution_trades['return'].mean()*100:>6.3f}%")
    print(f"  Win rate: {(execution_trades['return'] > 0).mean()*100:>5.1f}%")
    print(f"  Total return (compound): {((1 + execution_trades['return']).prod() - 1)*100:>6.2f}%")
    
    # Save to CSV
    csv_filename = 'execution_trades_bollinger.csv'
    execution_trades.to_csv(csv_filename, index=False)
    print(f"\n✅ Saved {len(execution_trades)} trades to {csv_filename}")