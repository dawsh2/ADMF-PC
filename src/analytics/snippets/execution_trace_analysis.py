# Execution trace analysis for comparing with analysis notebook
# Add this cell to trade_analysis.ipynb after loading the execution traces

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("\n" + "="*120)
print("EXECUTION ENGINE TRADE-BY-TRADE ANALYSIS")
print("="*120)

# Function to extract trades from execution traces
def extract_execution_trades(fills_df, positions_df, orders_df=None, max_trades=100):
    """
    Extract trades from execution engine traces.
    
    Args:
        fills_df: DataFrame with fill events
        positions_df: DataFrame with position events 
        orders_df: Optional DataFrame with order events
        max_trades: Maximum number of trades to analyze
        
    Returns:
        DataFrame with trade details
    """
    # Sort by timestamp
    fills_df = fills_df.sort_values('timestamp').reset_index(drop=True)
    positions_df = positions_df.sort_values('timestamp').reset_index(drop=True)
    
    trades = []
    current_positions = {}
    
    # Process fills to build trades
    for idx, fill in fills_df.iterrows():
        symbol = fill['symbol']
        
        # Check if this is an entry or exit
        if symbol not in current_positions or current_positions[symbol] is None:
            # New position entry
            current_positions[symbol] = {
                'entry_time': fill['timestamp'],
                'entry_price': fill['price'],
                'entry_fill_id': fill.get('fill_id', idx),
                'quantity': fill['quantity'],
                'side': fill['side'],
                'direction': 'LONG' if fill['side'] == 'BUY' else 'SHORT',
                'trade_num': len(trades) + 1
            }
        else:
            # Position exit
            entry = current_positions[symbol]
            
            # Calculate return
            entry_price = float(entry['entry_price'])
            exit_price = float(fill['price'])
            
            if entry['direction'] == 'LONG':
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
                
            # Apply execution cost (1 bp)
            net_return = raw_return - 0.0001
            
            # Determine exit type from fill metadata or order metadata
            exit_type = 'signal'  # default
            
            # Check fill metadata
            if 'exit_type' in fill:
                exit_type = fill['exit_type']
            elif 'metadata' in fill and isinstance(fill['metadata'], dict):
                exit_type = fill['metadata'].get('exit_type', 'signal')
            
            # Check if we have corresponding position data
            position_exit = positions_df[
                (positions_df['symbol'] == symbol) & 
                (positions_df['timestamp'] >= fill['timestamp'])
            ].iloc[0] if len(positions_df[
                (positions_df['symbol'] == symbol) & 
                (positions_df['timestamp'] >= fill['timestamp'])
            ]) > 0 else None
            
            if position_exit is not None and 'exit_type' in position_exit:
                exit_type = position_exit['exit_type']
            
            # Map exit types
            exit_type_map = {
                'stop_loss': 'stop',
                'take_profit': 'target',
                'signal': 'signal',
                'eod': 'signal'
            }
            exit_type = exit_type_map.get(exit_type, exit_type)
            
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
                'raw_return': raw_return,
                'entry_fill_id': entry['entry_fill_id'],
                'exit_fill_id': fill.get('fill_id', idx)
            }
            
            # Calculate stop and target prices (0.075% stop, 0.1% target)
            if entry['direction'] == 'LONG':
                trade['stop'] = entry_price * 0.99925  # 0.075% below
                trade['target'] = entry_price * 1.001   # 0.1% above
            else:
                trade['stop'] = entry_price * 1.00075   # 0.075% above
                trade['target'] = entry_price * 0.999    # 0.1% below
            
            # Calculate bars in trade (if we have bar timestamps)
            if 'bar_index' in fill and 'bar_index' in entry:
                trade['bars'] = fill['bar_index'] - entry.get('bar_index', 0)
            else:
                # Estimate from time difference (5-minute bars)
                time_diff = (fill['timestamp'] - entry['entry_time']).total_seconds() / 60
                trade['bars'] = max(1, int(time_diff / 5))
            
            trades.append(trade)
            current_positions[symbol] = None
            
            if len(trades) >= max_trades:
                break
    
    return pd.DataFrame(trades)

# Extract trades from execution traces
print("\nExtracting trades from execution traces...")

# Get fills and positions DataFrames
if 'fills_df' in globals() and 'positions_df' in globals():
    execution_trades = extract_execution_trades(
        fills_df, 
        positions_df,
        orders_df if 'orders_df' in globals() else None,
        max_trades=100
    )
    
    print(f"Found {len(execution_trades)} trades")
    
    # Display first 30 trades
    print(f"\nFirst 30 trades from EXECUTION ENGINE:")
    print("-"*120)
    print(f"{'#':>3} {'Entry Time':>20} {'Dir':>5} {'Entry':>8} {'Stop':>8} {'Target':>8} "
          f"{'Exit':>6} {'Exit$':>8} {'Ret%':>8} {'Bars':>5}")
    print("-"*120)
    
    for _, t in execution_trades.head(30).iterrows():
        print(f"{t['num']:>3} {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>20} "
              f"{t['dir']:>5} {t['entry_price']:>8.2f} {t['stop']:>8.2f} {t['target']:>8.2f} "
              f"{t['exit_type']:>6} {t['exit_price']:>8.2f} {t['return']*100:>7.3f}% {t.get('bars', 0):>5}")
    
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
    
    # Save to CSV for comparison
    csv_filename = 'execution_trades_bollinger.csv'
    execution_trades.to_csv(csv_filename, index=False)
    print(f"\n✅ Saved {len(execution_trades)} trades to {csv_filename}")
    
    # Load analysis trades if available for direct comparison
    analysis_csv = 'analysis_trades_5edc4365.csv'
    if Path(analysis_csv).exists():
        print(f"\n" + "="*120)
        print("DIRECT COMPARISON: ANALYSIS vs EXECUTION")
        print("="*120)
        
        analysis_trades = pd.read_csv(analysis_csv)
        analysis_trades['entry_time'] = pd.to_datetime(analysis_trades['entry_time'])
        analysis_trades['exit_time'] = pd.to_datetime(analysis_trades['exit_time'])
        
        # Compare first 10 trades side by side
        print("\nFirst 10 trades comparison:")
        print("-"*120)
        print(f"{'#':>3} {'ANALYSIS':^40} | {'EXECUTION':^40}")
        print(f"{'':>3} {'Entry Time':>16} {'Exit':>6} {'Ret%':>8} | "
              f"{'Entry Time':>16} {'Exit':>6} {'Ret%':>8}")
        print("-"*120)
        
        for i in range(min(10, len(analysis_trades), len(execution_trades))):
            a = analysis_trades.iloc[i]
            e = execution_trades.iloc[i]
            
            print(f"{i+1:>3} {a['entry_time'].strftime('%m-%d %H:%M'):>16} "
                  f"{a['exit_type']:>6} {a['return']*100:>7.2f}% | "
                  f"{e['entry_time'].strftime('%m-%d %H:%M'):>16} "
                  f"{e['exit_type']:>6} {e['return']*100:>7.2f}%")
        
        # Check if entry times match
        print("\n📊 Entry Time Matching (first 20 trades):")
        for i in range(min(20, len(analysis_trades), len(execution_trades))):
            a_time = analysis_trades.iloc[i]['entry_time']
            e_time = execution_trades.iloc[i]['entry_time']
            match = "✓" if abs((a_time - e_time).total_seconds()) < 60 else "✗"
            if match == "✗":
                print(f"Trade {i+1}: {match} Analysis: {a_time} | Execution: {e_time}")
        
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
    print("ERROR: fills_df and positions_df not found in notebook context")
    print("Please ensure you have loaded the execution traces before running this cell")
    print("\nExpected DataFrames:")
    print("- fills_df: DataFrame with fill events from execution")
    print("- positions_df: DataFrame with position events")
    print("- orders_df (optional): DataFrame with order events")