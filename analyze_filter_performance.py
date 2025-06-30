"""Analyze performance of each filtered Keltner strategy"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_57cd04b8")

# Load SPY data for prices
spy_data = pd.read_csv("./data/SPY.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.columns = spy_data.columns.str.lower()
spy_data = spy_data.set_index('timestamp')

strategies = [
    ('keltner_baseline', 'No filter', 193386),
    ('keltner_rsi50', 'RSI < 50', 4074),
    ('keltner_rsi70', 'RSI < 70', 32810),
    ('keltner_directional', '(Long: RSI<40) or (Short: RSI>60)', 34992)
]

results = []

for strategy_name, filter_desc, file_size in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Filter: {filter_desc}")
    print(f"File size: {file_size:,} bytes")
    print("="*60)
    
    # Load signals
    signal_file = workspace / f"traces/SPY_1m/signals/mean_reversion/SPY_{strategy_name}.parquet"
    if not signal_file.exists():
        print(f"File not found: {signal_file}")
        continue
        
    signals_df = pd.read_parquet(signal_file)
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts']).dt.tz_convert('UTC')
    signals_df = signals_df.sort_values('timestamp')
    
    print(f"Signal changes: {len(signals_df):,}")
    
    # Simple backtest
    positions = []
    current_position = None
    
    for _, signal in signals_df.iterrows():
        ts = signal['timestamp']
        signal_value = signal['val']
        
        # Get price
        if ts in spy_data.index:
            price = spy_data.loc[ts, 'close']
        else:
            price = signal['px']  # Use signal price if market price missing
        
        if signal_value != 0 and current_position is None:
            # Enter position
            current_position = {
                'entry_time': ts,
                'entry_price': price,
                'direction': 1 if signal_value > 0 else -1
            }
        elif signal_value == 0 and current_position is not None:
            # Exit position
            ret = (price - current_position['entry_price']) / current_position['entry_price']
            if current_position['direction'] < 0:
                ret = -ret
                
            current_position['exit_time'] = ts
            current_position['exit_price'] = price
            current_position['return'] = ret
            current_position['bps'] = ret * 10000
            current_position['duration_min'] = (ts - current_position['entry_time']).total_seconds() / 60
            
            positions.append(current_position)
            current_position = None
        elif signal_value != 0 and current_position is not None and np.sign(signal_value) != current_position['direction']:
            # Close and reverse
            ret = (price - current_position['entry_price']) / current_position['entry_price']
            if current_position['direction'] < 0:
                ret = -ret
                
            current_position['exit_time'] = ts
            current_position['exit_price'] = price
            current_position['return'] = ret
            current_position['bps'] = ret * 10000
            current_position['duration_min'] = (ts - current_position['entry_time']).total_seconds() / 60
            
            positions.append(current_position)
            
            # Open new position
            current_position = {
                'entry_time': ts,
                'entry_price': price,
                'direction': 1 if signal_value > 0 else -1
            }
    
    if len(positions) > 0:
        trades_df = pd.DataFrame(positions)
        
        # Calculate metrics
        total_trades = len(trades_df)
        avg_bps = trades_df['bps'].mean()
        win_rate = (trades_df['bps'] > 0).mean() * 100
        
        # Trading days
        trading_days = (signals_df['timestamp'].max() - signals_df['timestamp'].min()).days
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        
        # Calculate per-direction stats
        longs = trades_df[trades_df['direction'] > 0]
        shorts = trades_df[trades_df['direction'] < 0]
        
        long_bps = longs['bps'].mean() if len(longs) > 0 else 0
        short_bps = shorts['bps'].mean() if len(shorts) > 0 else 0
        
        # After costs
        cost_bps = 1.0
        net_bps = avg_bps - cost_bps
        annual_return = net_bps * trades_per_day * 252 / 100
        
        # Store results
        result = {
            'strategy': strategy_name,
            'filter': filter_desc,
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'avg_bps': avg_bps,
            'net_bps': net_bps,
            'win_rate': win_rate,
            'annual_return': annual_return,
            'avg_duration_min': trades_df['duration_min'].mean(),
            'long_trades': len(longs),
            'long_bps': long_bps,
            'short_trades': len(shorts),
            'short_bps': short_bps,
            'best_trade_bps': trades_df['bps'].max(),
            'worst_trade_bps': trades_df['bps'].min(),
            'std_bps': trades_df['bps'].std()
        }
        results.append(result)
        
        # Print summary
        print(f"\nPerformance Summary:")
        print(f"  Total trades: {total_trades}")
        print(f"  Trades per day: {trades_per_day:.2f}")
        print(f"  Average bps: {avg_bps:.2f}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Net bps (after 1bp cost): {net_bps:.2f}")
        print(f"  Annual return: {annual_return:.1f}%")
        print(f"  Avg duration: {trades_df['duration_min'].mean():.1f} minutes")
        print(f"\n  Longs: {len(longs)} trades, {long_bps:.2f} bps avg")
        print(f"  Shorts: {len(shorts)} trades, {short_bps:.2f} bps avg")
    else:
        print("\nNo completed trades")

# Create comparison table
if results:
    print("\n" + "="*80)
    print("FILTER COMPARISON SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    
    # Sort by net bps
    df = df.sort_values('net_bps', ascending=False)
    
    print("\nAll Strategies:")
    print(df[['strategy', 'filter', 'trades_per_day', 'avg_bps', 'net_bps', 'win_rate', 'annual_return']].to_string(index=False))
    
    # Find strategies meeting criteria
    good_strategies = df[(df['net_bps'] > 0) & (df['trades_per_day'] >= 2)]
    
    if len(good_strategies) > 0:
        print("\n🎯 STRATEGIES MEETING CRITERIA (positive net bps, >=2 trades/day):")
        print(good_strategies[['strategy', 'filter', 'trades_per_day', 'avg_bps', 'net_bps', 'annual_return']].to_string(index=False))
    
    # Save results
    df.to_csv('keltner_filter_performance.csv', index=False)
    print("\nDetailed results saved to: keltner_filter_performance.csv")