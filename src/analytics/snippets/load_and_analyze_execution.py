# Load execution traces and analyze trades
# Add this to your trade_analysis.ipynb

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime

# Set the results directory
results_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/latest')
print(f"Loading execution traces from: {results_dir}")

# Helper function to expand sparse metadata
def expand_metadata(df, metadata_col='metadata', prefix=''):
    """Expand nested metadata into columns."""
    if metadata_col in df.columns and len(df) > 0:
        # Extract metadata fields
        try:
            metadata_df = pd.json_normalize(df[metadata_col])
            # Add prefix to avoid column conflicts
            if prefix:
                metadata_df.columns = [f"{prefix}{col}" for col in metadata_df.columns]
            # Drop the original metadata column and combine
            result = pd.concat([df.drop(columns=[metadata_col]), metadata_df], axis=1)
            return result
        except Exception as e:
            print(f"Warning: Could not expand metadata: {e}")
            return df
    return df

# Load fills
fills_path = results_dir / 'traces/execution/fills'
fills_files = list(fills_path.glob('*.parquet'))
if fills_files:
    fills_df = pd.concat([pd.read_parquet(f) for f in fills_files], ignore_index=True)
    # Convert ts to timestamp
    fills_df['timestamp'] = pd.to_datetime(fills_df['ts'])
    # Expand metadata
    fills_df = expand_metadata(fills_df, prefix='fill_')
    fills_df = fills_df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(fills_df)} fills")
    print(f"   Columns: {fills_df.columns.tolist()[:10]}...")
else:
    print("❌ No fills found")
    fills_df = pd.DataFrame()

# Load positions separately and handle metadata carefully
positions_dfs = []

# Load position opens
positions_open_path = results_dir / 'traces/portfolio/positions_open'
open_files = list(positions_open_path.glob('*.parquet'))
if open_files:
    opens_df = pd.concat([pd.read_parquet(f) for f in open_files], ignore_index=True)
    opens_df['timestamp'] = pd.to_datetime(opens_df['ts'])
    opens_df['position_type'] = 'open'
    # Check what columns we have before expanding
    print(f"Position open columns before expansion: {opens_df.columns.tolist()}")
    positions_dfs.append(opens_df)

# Load position closes
positions_close_path = results_dir / 'traces/portfolio/positions_close'
close_files = list(positions_close_path.glob('*.parquet'))
if close_files:
    closes_df = pd.concat([pd.read_parquet(f) for f in close_files], ignore_index=True)
    closes_df['timestamp'] = pd.to_datetime(closes_df['ts'])
    closes_df['position_type'] = 'close'
    # Check what columns we have before expanding
    print(f"Position close columns before expansion: {closes_df.columns.tolist()}")
    positions_dfs.append(closes_df)

if positions_dfs:
    # Combine position dataframes
    positions_df = pd.concat(positions_dfs, ignore_index=True, sort=False)
    
    # Now expand metadata if it exists
    if 'metadata' in positions_df.columns:
        # Extract specific fields we need from metadata
        metadata_records = []
        for idx, row in positions_df.iterrows():
            meta = row.get('metadata', {})
            if isinstance(meta, dict):
                record = {
                    'symbol': meta.get('symbol', row.get('sym', '')),
                    'quantity': meta.get('quantity', 0),
                    'average_price': meta.get('average_price', 0),
                    'entry_price': meta.get('entry_price', 0),
                    'exit_price': meta.get('exit_price', 0),
                    'exit_type': meta.get('exit_type', 'signal'),
                    'exit_reason': meta.get('exit_reason', ''),
                    'realized_pnl': meta.get('realized_pnl', 0),
                    'strategy_id': meta.get('strategy_id', '')
                }
            else:
                record = {}
            metadata_records.append(record)
        
        # Add extracted fields to dataframe
        metadata_df = pd.DataFrame(metadata_records)
        for col in metadata_df.columns:
            if col not in positions_df.columns:
                positions_df[col] = metadata_df[col]
    
    positions_df = positions_df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(positions_df)} position events")
    print(f"   Columns: {positions_df.columns.tolist()}")
else:
    print("❌ No positions found")
    positions_df = pd.DataFrame()

# Load orders
orders_path = results_dir / 'traces/portfolio/orders'
orders_files = list(orders_path.glob('*.parquet'))
if orders_files:
    orders_df = pd.concat([pd.read_parquet(f) for f in orders_files], ignore_index=True)
    orders_df['timestamp'] = pd.to_datetime(orders_df['ts'])
    orders_df = expand_metadata(orders_df, prefix='order_')
    orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(orders_df)} orders")
else:
    print("❌ No orders found")
    orders_df = pd.DataFrame()

# Now run the analysis
print("\n" + "="*120)
print("EXECUTION ENGINE TRADE-BY-TRADE ANALYSIS")
print("="*120)

# Extract trades from fills
def extract_trades_from_traces(fills_df, positions_df=None, max_trades=100):
    """Extract trades from fill events."""
    
    trades = []
    current_positions = {}
    
    # Sort fills by timestamp
    fills_df = fills_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Processing {len(fills_df)} fills...")
    
    # Process fills to build trades
    for idx, fill in fills_df.iterrows():
        if len(trades) >= max_trades:
            break
            
        # Get symbol from various possible fields
        symbol = fill.get('symbol', fill.get('sym', fill.get('fill_symbol', 'SPY')))
        side = fill.get('side', fill.get('fill_side', ''))
        price = float(fill.get('price', fill.get('fill_price', 0)))
        quantity = float(fill.get('quantity', fill.get('fill_quantity', 0)))
        
        # Check if this is an entry or exit based on current positions
        if symbol not in current_positions or current_positions[symbol] is None:
            # New position entry
            if side in ['BUY', 'buy', 'LONG', 'long']:
                direction = 'LONG'
            else:
                direction = 'SHORT'
                
            current_positions[symbol] = {
                'entry_time': fill['timestamp'],
                'entry_price': price,
                'entry_idx': idx,
                'quantity': quantity,
                'side': side,
                'direction': direction,
                'trade_num': len(trades) + 1
            }
        else:
            # Position exit
            entry = current_positions[symbol]
            
            # Determine exit type from metadata
            exit_type = 'signal'  # default
            
            # Check various metadata fields
            if 'fill_exit_type' in fill:
                exit_type = fill['fill_exit_type']
            elif 'fill_metadata.exit_type' in fill:
                exit_type = fill['fill_metadata.exit_type']
            elif 'fill_order_metadata.exit_type' in fill:
                exit_type = fill['fill_order_metadata.exit_type']
            
            # Map exit types
            exit_type_map = {
                'stop_loss': 'stop',
                'take_profit': 'target',
                'signal': 'signal',
                'eod': 'signal'
            }
            exit_type = exit_type_map.get(exit_type, exit_type)
            
            # Calculate returns
            entry_price = float(entry['entry_price'])
            exit_price = price
            
            if entry['direction'] == 'LONG':
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
                
            # Apply execution cost (1 bp)
            net_return = raw_return - 0.0001
            
            trade = {
                'num': entry['trade_num'],
                'entry_time': entry['entry_time'],
                'entry_price': entry_price,
                'exit_time': fill['timestamp'],
                'exit_price': exit_price,
                'dir': entry['direction'],
                'quantity': abs(float(entry['quantity'])),
                'exit_type': exit_type,
                'return': net_return,
                'raw_return': raw_return
            }
            
            # Calculate stop and target prices (0.075% stop, 0.1% target)
            if entry['direction'] == 'LONG':
                trade['stop'] = entry_price * 0.99925  # 0.075% below
                trade['target'] = entry_price * 1.001   # 0.1% above
            else:
                trade['stop'] = entry_price * 1.00075   # 0.075% above
                trade['target'] = entry_price * 0.999    # 0.1% below
            
            # Calculate bars in trade (5-minute bars)
            time_diff = (fill['timestamp'] - entry['entry_time']).total_seconds() / 60
            trade['bars'] = max(1, int(time_diff / 5))
            
            trades.append(trade)
            current_positions[symbol] = None
    
    return pd.DataFrame(trades)

# Extract trades
execution_trades = extract_trades_from_traces(fills_df, positions_df, max_trades=100)

if len(execution_trades) > 0:
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
    print(f"\nExit Type Summary (first {len(execution_trades)} trades):")
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
    
    # Compare with analysis if available
    analysis_csv = 'analysis_trades_5edc4365.csv'
    if Path(analysis_csv).exists():
        print(f"\n" + "="*120)
        print("DIRECT COMPARISON: ANALYSIS vs EXECUTION")
        print("="*120)
        
        analysis_trades = pd.read_csv(analysis_csv)
        analysis_trades['entry_time'] = pd.to_datetime(analysis_trades['entry_time'])
        analysis_trades['exit_time'] = pd.to_datetime(analysis_trades['exit_time'])
        
        # Compare first 10 trades
        print("\nFirst 10 trades comparison:")
        print("-"*120)
        print(f"{'#':>3} {'ANALYSIS':^50} | {'EXECUTION':^50}")
        print(f"{'':>3} {'Entry Time':>16} {'Dir':>5} {'Exit':>6} {'Ret%':>8} | "
              f"{'Entry Time':>16} {'Dir':>5} {'Exit':>6} {'Ret%':>8}")
        print("-"*120)
        
        for i in range(min(10, len(analysis_trades), len(execution_trades))):
            a = analysis_trades.iloc[i]
            e = execution_trades.iloc[i]
            
            print(f"{i+1:>3} {a['entry_time'].strftime('%m-%d %H:%M'):>16} "
                  f"{a['dir']:>5} {a['exit_type']:>6} {a['return']*100:>7.2f}% | "
                  f"{e['entry_time'].strftime('%m-%d %H:%M'):>16} "
                  f"{e['dir']:>5} {e['exit_type']:>6} {e['return']*100:>7.2f}%")
        
        # Summary comparison
        print(f"\n📊 SUMMARY COMPARISON:")
        print(f"{'Metric':>20} {'Analysis':>15} {'Execution':>15} {'Difference':>15}")
        print("-"*65)
        
        metrics = [
            ('Total trades', len(analysis_trades), len(execution_trades)),
            ('Stop exits', 
             len(analysis_trades[analysis_trades['exit_type'] == 'stop']),
             len(execution_trades[execution_trades['exit_type'] == 'stop'])),
            ('Target exits',
             len(analysis_trades[analysis_trades['exit_type'] == 'target']),
             len(execution_trades[execution_trades['exit_type'] == 'target'])),
            ('Signal exits',
             len(analysis_trades[analysis_trades['exit_type'] == 'signal']),
             len(execution_trades[execution_trades['exit_type'] == 'signal'])),
            ('Avg return %',
             analysis_trades['return'].mean() * 100,
             execution_trades['return'].mean() * 100),
            ('Win rate %',
             (analysis_trades['return'] > 0).mean() * 100,
             (execution_trades['return'] > 0).mean() * 100),
            ('Total return %',
             ((1 + analysis_trades['return']).prod() - 1) * 100,
             ((1 + execution_trades['return']).prod() - 1) * 100)
        ]
        
        for name, a_val, e_val in metrics:
            if isinstance(a_val, float):
                diff = e_val - a_val
                print(f"{name:>20} {a_val:>14.2f} {e_val:>14.2f} {diff:>+14.2f}")
            else:
                diff = e_val - a_val
                print(f"{name:>20} {a_val:>14} {e_val:>14} {diff:>+14}")
                
else:
    print("No trades found in execution traces")
    
# Debug: Show sample of position data
if len(positions_df) > 0:
    print("\n📊 DEBUG: Sample position events")
    cols_to_show = ['timestamp', 'position_type', 'symbol', 'quantity', 'average_price', 'exit_type', 'exit_reason']
    available_cols = [col for col in cols_to_show if col in positions_df.columns]
    print(f"Available columns: {available_cols}")
    if available_cols:
        print(positions_df[available_cols].head(10))