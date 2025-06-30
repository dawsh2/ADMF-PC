#!/usr/bin/env python3
"""Deep dive into catastrophic losing trades"""

import pandas as pd
import numpy as np
from pathlib import Path

# Focus on test set P=22, M=0.5
trace_file = Path("config/keltner/test_top10/results/20250622_220133/traces/mean_reversion/SPY_5m_kb_p22_m05.parquet")
df_trace = pd.read_parquet(trace_file)

# Load SPY data with more detail
spy_data = pd.read_csv("data/SPY_5m.csv")
spy_data['datetime'] = pd.to_datetime(spy_data['timestamp'])
spy_data['date'] = spy_data['datetime'].dt.date
spy_data['time'] = spy_data['datetime'].dt.time

print("CATASTROPHIC TRADE ANALYSIS - P=22, M=0.5 TEST SET")
print("="*80)

# Reconstruct trades with full detail
trades = []
in_trade = False

for i in range(len(df_trace) - 1):
    current_signal = df_trace.iloc[i]['val']
    current_idx = df_trace.iloc[i]['idx']
    
    if not in_trade and current_signal != 0:
        # Entry
        in_trade = True
        entry_idx = current_idx
        entry_price = spy_data.iloc[entry_idx]['close']
        entry_signal = current_signal
        entry_time = spy_data.iloc[entry_idx]['datetime']
        
    elif in_trade and current_signal == 0:
        # Exit
        in_trade = False
        exit_idx = df_trace.iloc[i]['idx']
        exit_price = spy_data.iloc[exit_idx]['close']
        exit_time = spy_data.iloc[exit_idx]['datetime']
        
        # Calculate return
        if entry_signal > 0:  # Long
            trade_return = (exit_price - entry_price) / entry_price
        else:  # Short
            trade_return = (entry_price - exit_price) / entry_price
        
        # Track the price path during the trade
        price_path = []
        for j in range(entry_idx, min(exit_idx + 1, len(spy_data))):
            price_path.append(spy_data.iloc[j]['close'])
        
        if entry_signal > 0:  # Long
            max_price = max(price_path)
            min_price = min(price_path)
            max_profit = (max_price - entry_price) / entry_price
            max_loss = (min_price - entry_price) / entry_price
        else:  # Short
            max_price = max(price_path)
            min_price = min(price_path)
            max_profit = (entry_price - min_price) / entry_price
            max_loss = (entry_price - max_price) / entry_price
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_bars': exit_idx - entry_idx,
            'direction': 'Long' if entry_signal > 0 else 'Short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_return,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'max_adverse_excursion': abs(max_loss),
            'entry_idx': entry_idx,
            'exit_idx': exit_idx
        })

df_trades = pd.DataFrame(trades)

# Find worst trades
worst_trades = df_trades.nsmallest(10, 'return')

print(f"\n10 WORST TRADES:")
print("-"*80)
print(f"{'#':<3} {'Entry Time':<20} {'Exit Time':<20} {'Dir':<5} {'Bars':<6} {'Return%':<8} {'MAE%':<8}")
print("-"*80)

for idx, (_, trade) in enumerate(worst_trades.iterrows(), 1):
    return_pct = trade['return'] * 100
    mae_pct = trade['max_adverse_excursion'] * 100
    print(f"{idx:<3} {str(trade['entry_time']):<20} {str(trade['exit_time']):<20} "
          f"{trade['direction']:<5} {trade['duration_bars']:<6} {return_pct:<8.2f} "
          f"{mae_pct:<8.2f}")

# Analyze patterns in bad trades
print(f"\nPATTERNS IN LOSING TRADES (worse than -0.5%):")
print("-"*80)

bad_trades = df_trades[df_trades['return'] < -0.005]
print(f"Total bad trades: {len(bad_trades)}")
print(f"Average duration: {bad_trades['duration_bars'].mean():.1f} bars vs {df_trades['duration_bars'].mean():.1f} overall")
bad_mae = bad_trades['max_adverse_excursion'].mean() * 100
all_mae = df_trades['max_adverse_excursion'].mean() * 100
print(f"Average MAE: {bad_mae:.2f}% vs {all_mae:.2f}% overall")

# Time of day analysis
bad_trades['entry_hour'] = pd.to_datetime(bad_trades['entry_time']).dt.hour
print(f"\nEntry hour distribution of bad trades:")
hour_counts = bad_trades['entry_hour'].value_counts().sort_index()
for hour, count in hour_counts.items():
    print(f"  {hour}:00 - {count} trades")

# Check if trades are respecting Keltner bands
print(f"\nKELTNER BAND ANALYSIS FOR WORST TRADES:")
print("-"*80)

# Calculate Keltner bands for the worst trade periods
for idx, trade in worst_trades.head(5).iterrows():
    start_idx = max(0, trade['entry_idx'] - 22)  # Need 22 bars for calculation
    end_idx = trade['exit_idx'] + 1
    
    trade_data = spy_data.iloc[start_idx:end_idx].copy()
    
    # Calculate Keltner bands (Period=22, Multiplier=0.5)
    trade_data['sma'] = trade_data['close'].rolling(22).mean()
    trade_data['atr'] = trade_data[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                     abs(x['high'] - x['close']), 
                     abs(x['low'] - x['close'])), axis=1
    ).rolling(22).mean()
    
    trade_data['upper_band'] = trade_data['sma'] + 0.5 * trade_data['atr']
    trade_data['lower_band'] = trade_data['sma'] - 0.5 * trade_data['atr']
    
    entry_row = trade_data.iloc[trade['entry_idx'] - start_idx]
    
    trade_return_pct = trade['return'] * 100
    print(f"\nTrade {idx+1} ({trade_return_pct:.2f}% loss):")
    print(f"  Entry: {trade['entry_time']}")
    if trade['direction'] == 'Long':
        print(f"  Entry price: ${trade['entry_price']:.2f} (Lower band: ${entry_row['lower_band']:.2f})")
        dist_from_band = ((trade['entry_price'] - entry_row['lower_band']) / entry_row['lower_band'] * 100)
        print(f"  Distance from band: {dist_from_band:.2f}%")
    else:
        print(f"  Entry price: ${trade['entry_price']:.2f} (Upper band: ${entry_row['upper_band']:.2f})")
        dist_from_band = ((entry_row['upper_band'] - trade['entry_price']) / trade['entry_price'] * 100)
        print(f"  Distance from band: {dist_from_band:.2f}%")

# Exit analysis
print(f"\nEXIT TIMING ANALYSIS:")
print("-"*80)

# For each bad trade, find when it first hit various stop levels
for stop_level in [0.25, 0.5, 1.0]:
    stopped_count = 0
    total_saved = 0
    
    for _, trade in bad_trades.iterrows():
        # Check each bar during the trade
        for bar_idx in range(trade['entry_idx'], trade['exit_idx']):
            if bar_idx < len(spy_data):
                current_price = spy_data.iloc[bar_idx]['close']
                
                if trade['direction'] == 'Long':
                    current_loss = (current_price - trade['entry_price']) / trade['entry_price']
                else:
                    current_loss = (trade['entry_price'] - current_price) / trade['entry_price']
                
                if current_loss < -stop_level/100:
                    # Would have been stopped
                    stopped_count += 1
                    saved = (trade['return'] - (-stop_level/100)) * 100
                    total_saved += saved
                    break
    
    print(f"{stop_level}% stop: Would exit {stopped_count}/{len(bad_trades)} bad trades, "
          f"saving {total_saved:.2f}% total return")