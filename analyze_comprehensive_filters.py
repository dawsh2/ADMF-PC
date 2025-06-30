"""Analyze comprehensive filter performance"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_75d250ab")

# Load SPY data for prices
spy_data = pd.read_csv("./data/SPY.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.columns = spy_data.columns.str.lower()
spy_data = spy_data.set_index('timestamp')

strategies = [
    ('kb_baseline', 'No filter'),
    ('kb_rsi_entry', 'Exit allowed + RSI<70'),
    ('kb_volume', 'Exit allowed + Volume>1.2x'),
    ('kb_volatility', 'Exit allowed + Vol%>60'),
    ('kb_vwap', 'Exit allowed + VWAP<0.2%'),
    ('kb_rsi_volume', 'Exit allowed + RSI<70 & Vol>1.1x'),
    ('kb_rsi_volatility', 'Exit allowed + RSI<70 & Vol%>50'),
    ('kb_all_strict', 'Exit allowed + RSI<50 & Vol>1.5x & Vol%>70'),
    ('kb_tight_1p5', 'Tighter bands (1.5x)'),
    ('kb_wide_2p5', 'Wider bands (2.5x)'),
    ('kb_fast_10', 'Faster period (10)'),
    ('kb_slow_50', 'Slower period (50)')
]

results = []

for strategy_name, description in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Description: {description}")
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
    print(f"File size: {signal_file.stat().st_size:,} bytes")
    
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
            'description': description,
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'avg_bps': avg_bps,
            'net_bps': net_bps,
            'win_rate': win_rate,
            'annual_return': annual_return,
            'avg_duration_min': trades_df['duration_min'].mean(),
            'median_duration_min': trades_df['duration_min'].median(),
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
        print(f"  Avg duration: {trades_df['duration_min'].mean():.1f} min (median: {trades_df['duration_min'].median():.1f} min)")
        print(f"\n  Longs: {len(longs)} trades, {long_bps:.2f} bps avg")
        print(f"  Shorts: {len(shorts)} trades, {short_bps:.2f} bps avg")
        
        # Check for exit signal issues
        signal_values = signals_df['val'].values
        signal_changes = np.diff(signal_values, prepend=signal_values[0])
        exits = ((signal_changes != 0) & (signals_df['val'] == 0)).sum()
        print(f"\n  Exit signals: {exits}")
    else:
        print("\nNo completed trades")

# Create comparison table
if results:
    print("\n" + "="*100)
    print("COMPREHENSIVE FILTER COMPARISON")
    print("="*100)
    
    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    
    # Sort by net bps
    df = df.sort_values('net_bps', ascending=False)
    
    print("\nAll Strategies (sorted by net bps):")
    print(df[['strategy', 'description', 'trades_per_day', 'avg_bps', 'net_bps', 'win_rate', 'annual_return', 'avg_duration_min']].to_string(index=False))
    
    # Find strategies meeting criteria
    good_strategies = df[(df['net_bps'] > 0) & (df['trades_per_day'] >= 2)]
    
    if len(good_strategies) > 0:
        print("\n🎯 STRATEGIES MEETING CRITERIA (positive net bps, >=2 trades/day):")
        print(good_strategies[['strategy', 'description', 'trades_per_day', 'avg_bps', 'net_bps', 'annual_return']].to_string(index=False))
    else:
        print("\n❌ No strategies meet the criteria (positive net bps with >=2 trades/day)")
    
    # Best parameter variations
    print("\n📊 PARAMETER OPTIMIZATION INSIGHTS:")
    param_strategies = df[df['strategy'].str.contains('kb_tight|kb_wide|kb_fast|kb_slow|kb_baseline')]
    print(param_strategies[['strategy', 'description', 'trades_per_day', 'avg_bps', 'net_bps']].to_string(index=False))
    
    # Save results
    df.to_csv('keltner_comprehensive_results.csv', index=False)
    print("\nDetailed results saved to: keltner_comprehensive_results.csv")