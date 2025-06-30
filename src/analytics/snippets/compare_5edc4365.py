# Trade-by-trade analysis for strategy 5edc4365
# This file is meant to be run with %run in the analysis notebook

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*120)
print("DETAILED TRADE-BY-TRADE ANALYSIS FOR STRATEGY 5edc4365")
print("="*120)

# Check if strategy_index exists, if not load it
if 'strategy_index' not in globals():
    print("Loading strategy index...")
    strategy_index_path = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448/strategy_index.parquet')
    strategy_index = pd.read_parquet(strategy_index_path)
    print(f"Loaded {len(strategy_index)} strategies")

# Check if run_dir is set
if 'run_dir' not in globals():
    run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448')
    print(f"Set run_dir to: {run_dir}")

# Check if market_data exists, if not try to load it
if 'market_data' not in globals():
    print("Loading market data...")
    market_data_path = Path('/Users/daws/ADMF-PC/data/SPY_5m.csv')
    if market_data_path.exists():
        market_data = pd.read_csv(market_data_path)
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        market_data = market_data.sort_values('timestamp')
        print(f"Loaded market data: {len(market_data)} bars")
    else:
        print(f"ERROR: Market data not found at {market_data_path}")
        print("Please ensure market_data is loaded in your notebook before running this script")

# Get strategy info
strategy_hash = '5edc43651004'
strategy_matches = strategy_index[strategy_index['strategy_hash'] == strategy_hash]
if len(strategy_matches) == 0:
    print(f"ERROR: Strategy {strategy_hash} not found in strategy index!")
    print("Available strategy hashes:")
    print(strategy_index['strategy_hash'].head(10).tolist())
else:
    strategy_info = strategy_matches.iloc[0]
    trace_path = strategy_info['trace_path']

    print(f"\nStrategy: {strategy_hash[:8]}")
    print(f"Parameters: period={strategy_info['period']}, std_dev={strategy_info['std_dev']}")
    print(f"Stop/Target: 0.075% / 0.1%")

    # Load signals
    signals_path = run_dir / trace_path
    signals = pd.read_parquet(signals_path)
    signals['ts'] = pd.to_datetime(signals['ts'])

    # Merge with market data
    df = market_data.merge(
        signals[['ts', 'val', 'px']], 
        left_on='timestamp', 
        right_on='ts', 
        how='left'
    )

    # Forward fill signals
    df['signal'] = df['val'].ffill().fillna(0)
    df['position'] = df['signal'].replace({0: 0, 1: 1, -1: -1})
    df['position_change'] = df['position'].diff().fillna(0)

    # Extract first 100 trades with stop/target analysis
    trades = []
    current_trade = None
    stop_pct = 0.075
    target_pct = 0.1
    execution_cost_bps = 1.0

    for idx, row in df.iterrows():
        if len(trades) >= 100 and current_trade is None:
            break
            
        # New position opened
        if row['position_change'] != 0 and row['position'] != 0:
            if current_trade is None:
                current_trade = {
                    'trade_num': len(trades) + 1,
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'direction': 'LONG' if row['position'] == 1 else 'SHORT',
                    'entry_idx': idx,
                    'entry_signal': row['signal']
                }
                
        # Position closed
        elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
            exit_price = row['px'] if pd.notna(row['px']) else row['close']
            
            # Calculate stop and target prices
            entry_price = current_trade['entry_price']
            if current_trade['direction'] == 'LONG':
                stop_price = entry_price * (1 - stop_pct/100)
                target_price = entry_price * (1 + target_pct/100)
            else:
                stop_price = entry_price * (1 + stop_pct/100)
                target_price = entry_price * (1 - target_pct/100)
            
            # Check bars for stop/target hits
            # IMPORTANT: Start from entry_idx+1 to avoid lookahead bias
            # We can't check the entry bar's high/low when we enter at close
            trade_bars = df.iloc[current_trade['entry_idx']+1:idx+1]
            actual_exit_price = exit_price
            actual_exit_time = row['timestamp']
            exit_type = 'signal'
            exit_bar = len(trade_bars) - 1
            
            for bar_idx, (_, bar) in enumerate(trade_bars.iterrows()):
                if current_trade['direction'] == 'LONG':
                    if bar['low'] <= stop_price:
                        actual_exit_price = stop_price
                        actual_exit_time = bar['timestamp']
                        exit_type = 'stop'
                        exit_bar = bar_idx
                        break
                    elif bar['high'] >= target_price:
                        actual_exit_price = target_price
                        actual_exit_time = bar['timestamp']
                        exit_type = 'target'
                        exit_bar = bar_idx
                        break
                else:  # SHORT
                    if bar['high'] >= stop_price:
                        actual_exit_price = stop_price
                        actual_exit_time = bar['timestamp']
                        exit_type = 'stop'
                        exit_bar = bar_idx
                        break
                    elif bar['low'] <= target_price:
                        actual_exit_price = target_price
                        actual_exit_time = bar['timestamp']
                        exit_type = 'target'
                        exit_bar = bar_idx
                        break
            
            # Calculate returns
            if current_trade['direction'] == 'LONG':
                raw_return = (actual_exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - actual_exit_price) / entry_price
            
            net_return = raw_return - execution_cost_bps / 10000
            
            trades.append({
                'num': current_trade['trade_num'],
                'entry_time': current_trade['entry_time'],
                'entry_price': entry_price,
                'dir': current_trade['direction'],
                'stop': stop_price,
                'target': target_price,
                'exit_type': exit_type,
                'exit_price': actual_exit_price,
                'exit_time': actual_exit_time,
                'bars': exit_bar + 1,
                'return': net_return,
                'entry_signal': current_trade['entry_signal'],
                'exit_signal': row['signal']
            })
            
            current_trade = None
            if row['position'] != 0 and row['position_change'] != 0:
                current_trade = {
                    'trade_num': len(trades) + 1,
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'direction': 'LONG' if row['position'] == 1 else 'SHORT',
                    'entry_idx': idx,
                    'entry_signal': row['signal']
                }

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    # Display first 30 trades
    print(f"\nFirst 30 trades (of {len(trades_df)} total):")
    print("-"*120)
    print(f"{'#':>3} {'Entry Time':>20} {'Dir':>5} {'Entry':>8} {'Stop':>8} {'Target':>8} "
          f"{'Exit':>6} {'Exit$':>8} {'Ret%':>8} {'Bars':>5}")
    print("-"*120)

    for _, t in trades_df.head(30).iterrows():
        print(f"{t['num']:>3} {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>20} "
              f"{t['dir']:>5} {t['entry_price']:>8.2f} {t['stop']:>8.2f} {t['target']:>8.2f} "
              f"{t['exit_type']:>6} {t['exit_price']:>8.2f} {t['return']*100:>7.3f}% {t['bars']:>5}")

    # Summary
    exit_counts = trades_df['exit_type'].value_counts()
    print(f"\nExit Type Summary (first 100 trades):")
    for exit_type in ['stop', 'target', 'signal']:
        count = exit_counts.get(exit_type, 0)
        print(f"  {exit_type:>8}: {count:>4} ({count/len(trades_df)*100:>5.1f}%)")

    print(f"\nPerformance:")
    print(f"  Avg return per trade: {trades_df['return'].mean()*100:>6.3f}%")
    print(f"  Win rate: {(trades_df['return'] > 0).mean()*100:>5.1f}%")
    print(f"  Total return (compound): {((1 + trades_df['return']).prod() - 1)*100:>6.2f}%")

    # Save to CSV
    csv_filename = 'analysis_trades_5edc4365.csv'
    trades_df.to_csv(csv_filename, index=False)
    print(f"\n✅ Saved {len(trades_df)} trades to {csv_filename}")
    print("\nCompare this with your execution engine output to find discrepancies!")