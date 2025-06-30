#!/usr/bin/env python3
"""Check overnight and weekend holding in test set"""

import pandas as pd
from pathlib import Path

# Analyze all test strategies
test_dir = Path("config/keltner/test_top10/results/20250622_220133")
traces_dir = test_dir / "traces" / "mean_reversion"

# Load SPY data
spy_data = pd.read_csv("data/SPY_5m.csv")
spy_data['datetime'] = pd.to_datetime(spy_data['timestamp'])
spy_data['date'] = spy_data['datetime'].dt.date
spy_data['time'] = spy_data['datetime'].dt.time
spy_data['hour'] = spy_data['datetime'].dt.hour
spy_data['day_of_week'] = spy_data['datetime'].dt.dayofweek  # 0=Monday, 4=Friday

print("OVERNIGHT AND WEEKEND HOLDING ANALYSIS - TEST SET")
print("="*80)

# Check each strategy
for trace_file in sorted(traces_dir.glob("*.parquet")):
    strategy_name = trace_file.stem.replace('SPY_5m_', '')
    
    df_trace = pd.read_parquet(trace_file)
    
    # Reconstruct trades
    trades = []
    overnight_trades = 0
    weekend_trades = 0
    late_entries = 0
    
    in_trade = False
    for i in range(len(df_trace) - 1):
        current_signal = df_trace.iloc[i]['val']
        current_idx = df_trace.iloc[i]['idx']
        
        if not in_trade and current_signal != 0:
            # Entry
            in_trade = True
            entry_idx = current_idx
            entry_time = spy_data.iloc[entry_idx]['datetime']
            entry_hour = spy_data.iloc[entry_idx]['hour']
            entry_dow = spy_data.iloc[entry_idx]['day_of_week']
            entry_date = spy_data.iloc[entry_idx]['date']
            
            if entry_hour >= 19:  # 7 PM or later
                late_entries += 1
                
        elif in_trade and current_signal == 0:
            # Exit
            in_trade = False
            exit_idx = df_trace.iloc[i]['idx']
            exit_time = spy_data.iloc[exit_idx]['datetime']
            exit_date = spy_data.iloc[exit_idx]['date']
            exit_dow = spy_data.iloc[exit_idx]['day_of_week']
            
            # Check if held overnight
            if entry_date != exit_date:
                overnight_trades += 1
                
                # Check if held over weekend
                if entry_dow == 4 and exit_dow == 0:  # Friday to Monday
                    weekend_trades += 1
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'overnight': entry_date != exit_date,
                'weekend': entry_dow == 4 and exit_dow == 0
            })
    
    if trades:
        print(f"\n{strategy_name}:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Overnight holds: {overnight_trades} ({overnight_trades/len(trades)*100:.1f}%)")
        print(f"  Weekend holds: {weekend_trades} ({weekend_trades/len(trades)*100:.1f}%)")
        print(f"  Late entries (>=19:00): {late_entries} ({late_entries/len(trades)*100:.1f}%)")

# Now compare with training data
print("\n" + "="*80)
print("COMPARISON WITH TRAINING DATA")
print("="*80)

# Pick the same strategy from training (P=22, M=0.5)
training_trace = Path("config/keltner/results/20250622_215020/traces/keltner_bands/SPY_5m_compiled_strategy_209.parquet")
df_train = pd.read_parquet(training_trace)

# Quick analysis
train_overnight = 0
train_trades = 0
train_late = 0

in_trade = False
for i in range(len(df_train) - 1):
    current_signal = df_train.iloc[i]['val']
    current_idx = df_train.iloc[i]['idx']
    
    if not in_trade and current_signal != 0:
        in_trade = True
        entry_idx = current_idx
        if entry_idx < len(spy_data):
            entry_date = spy_data.iloc[entry_idx]['date']
            entry_hour = spy_data.iloc[entry_idx]['hour']
            if entry_hour >= 19:
                train_late += 1
            
    elif in_trade and current_signal == 0:
        in_trade = False
        exit_idx = df_train.iloc[i]['idx'] 
        train_trades += 1
        
        if entry_idx < len(spy_data) and exit_idx < len(spy_data):
            exit_date = spy_data.iloc[exit_idx]['date']
            if 'entry_date' in locals() and entry_date != exit_date:
                train_overnight += 1

print(f"\nTraining P=22, M=0.5:")
print(f"  Total trades: {train_trades}")
print(f"  Overnight holds: {train_overnight} ({train_overnight/train_trades*100:.1f}%)")
print(f"  Late entries: {train_late} ({train_late/train_trades*100:.1f}%)")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("\n1. Test set has MUCH higher overnight holding rates")
print("2. Many trades enter late in the day (after 7 PM)")
print("3. Weekend holds are particularly dangerous")
print("\nSUGGESTION: Add time-of-day filter to avoid entries after 3:30 PM")