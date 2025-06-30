"""Deep dive into trend performance by long/short direction"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal data
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"
signals = pd.read_parquet(signal_file)

# Load raw SPY data to calculate additional indicators
spy_data = pd.read_csv("./data/SPY_1m.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                         'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Calculate trend indicators
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()

# Calculate VWAP
spy_data['date'] = spy_data['timestamp'].dt.date
spy_data['typical_price'] = (spy_data['high'] + spy_data['low'] + spy_data['close']) / 3
spy_data['pv'] = spy_data['typical_price'] * spy_data['volume']
spy_data['cum_pv'] = spy_data.groupby('date')['pv'].cumsum()
spy_data['cum_volume'] = spy_data.groupby('date')['volume'].cumsum()
spy_data['vwap'] = spy_data['cum_pv'] / spy_data['cum_volume']

# Define trend conditions
spy_data['trend_up'] = (spy_data['close'] > spy_data['sma_50']) & (spy_data['sma_50'] > spy_data['sma_200'])
spy_data['trend_down'] = (spy_data['close'] < spy_data['sma_50']) & (spy_data['sma_50'] < spy_data['sma_200'])
spy_data['trend_neutral'] = ~(spy_data['trend_up'] | spy_data['trend_down'])

# VWAP position
spy_data['above_vwap'] = spy_data['close'] > spy_data['vwap']

# Calculate trades
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_idx = prev_signal['idx']
            exit_idx = curr_signal['idx']
            
            entry_conditions = spy_data.iloc[entry_idx] if entry_idx < len(spy_data) else None
            
            if entry_conditions is not None and not pd.isna(entry_conditions['sma_200']):
                entry_price = prev_signal['px']
                exit_price = curr_signal['px']
                signal_type = prev_signal['val']
                
                pct_return = (exit_price / entry_price - 1) * signal_type * 100
                
                trades.append({
                    'signal': signal_type,
                    'pct_return': pct_return,
                    'trend_up': entry_conditions['trend_up'],
                    'trend_down': entry_conditions['trend_down'],
                    'trend_neutral': entry_conditions['trend_neutral'],
                    'above_vwap': entry_conditions['above_vwap'],
                })

trades_df = pd.DataFrame(trades)

print("=== DETAILED BREAKDOWN: LONG vs SHORT BY TREND ===\n")

# Create detailed breakdown
for trend, trend_name in [('trend_up', 'UPTREND'), ('trend_down', 'DOWNTREND'), ('trend_neutral', 'NEUTRAL')]:
    trend_trades = trades_df[trades_df[trend]]
    if len(trend_trades) == 0:
        continue
        
    print(f"\n{trend_name} ({len(trend_trades)} total trades):")
    print("=" * 50)
    
    # Overall trend performance
    total_return = np.exp(np.log(1 + trend_trades['pct_return']/100).sum()) - 1
    print(f"Total return in {trend_name}: {total_return*100:.2f}%")
    
    # Break down by direction
    for signal, direction in [(1, 'LONG'), (-1, 'SHORT')]:
        direction_trades = trend_trades[trend_trades['signal'] == signal]
        if len(direction_trades) > 0:
            avg_return = direction_trades['pct_return'].mean()
            total_return = np.exp(np.log(1 + direction_trades['pct_return']/100).sum()) - 1
            win_rate = (direction_trades['pct_return'] > 0).mean()
            
            print(f"\n  {direction} trades in {trend_name}:")
            print(f"    Count: {len(direction_trades)}")
            print(f"    Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
            print(f"    Total return: {total_return*100:.2f}%")
            print(f"    Win rate: {win_rate:.1%}")

# VWAP analysis by direction
print("\n\n=== VWAP POSITION ANALYSIS BY DIRECTION ===\n")

# Analyze VWAP mean reversion vs trend following
vwap_scenarios = [
    ("VWAP Mean Reversion", [
        ((trades_df['above_vwap'] == True) & (trades_df['signal'] == -1), "Short when above VWAP"),
        ((trades_df['above_vwap'] == False) & (trades_df['signal'] == 1), "Long when below VWAP")
    ]),
    ("VWAP Trend Following", [
        ((trades_df['above_vwap'] == True) & (trades_df['signal'] == 1), "Long when above VWAP"),
        ((trades_df['above_vwap'] == False) & (trades_df['signal'] == -1), "Short when below VWAP")
    ])
]

for scenario_name, conditions in vwap_scenarios:
    print(f"\n{scenario_name}:")
    print("-" * 40)
    
    scenario_total = 0
    scenario_trades = 0
    
    for condition, desc in conditions:
        filtered = trades_df[condition]
        if len(filtered) > 0:
            avg_return = filtered['pct_return'].mean()
            total_return = np.exp(np.log(1 + filtered['pct_return']/100).sum()) - 1
            win_rate = (filtered['pct_return'] > 0).mean()
            
            scenario_total += total_return
            scenario_trades += len(filtered)
            
            print(f"\n  {desc}:")
            print(f"    Trades: {len(filtered)}")
            print(f"    Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
            print(f"    Total return: {total_return*100:.2f}%")
            print(f"    Win rate: {win_rate:.1%}")
    
    print(f"\n  Combined {scenario_name}: {scenario_trades} trades, {scenario_total*100:.2f}% total")

# Best combinations
print("\n\n=== OPTIMAL STRATEGY COMBINATIONS ===\n")

optimal_filters = [
    ("Short in any trend", trades_df['signal'] == -1),
    ("Short in uptrend (counter-trend)", (trades_df['trend_up']) & (trades_df['signal'] == -1)),
    ("Short in downtrend (with trend)", (trades_df['trend_down']) & (trades_df['signal'] == -1)),
    ("Long in uptrend only", (trades_df['trend_up']) & (trades_df['signal'] == 1)),
    ("Short when above VWAP", (trades_df['above_vwap']) & (trades_df['signal'] == -1)),
    ("Short in uptrend + above VWAP", (trades_df['trend_up']) & (trades_df['above_vwap']) & (trades_df['signal'] == -1)),
]

for desc, condition in optimal_filters:
    filtered = trades_df[condition]
    if len(filtered) > 0:
        avg_return = filtered['pct_return'].mean()
        total_return = np.exp(np.log(1 + filtered['pct_return']/100).sum()) - 1
        win_rate = (filtered['pct_return'] > 0).mean()
        
        print(f"\n{desc}:")
        print(f"  Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.1f}% of all)")
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Total return: {total_return*100:.2f}%")
        print(f"  Win rate: {win_rate:.1%}")