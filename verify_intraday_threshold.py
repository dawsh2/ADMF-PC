#!/usr/bin/env python3
"""Verify that intraday threshold is working correctly."""

import pandas as pd
import sys

# Check if we have the latest results
try:
    # Load the signals
    signals_df = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
    signals_df['ts'] = pd.to_datetime(signals_df['ts'])
    
    # Add time columns
    signals_df['hour'] = signals_df['ts'].dt.hour
    signals_df['minute'] = signals_df['ts'].dt.minute
    signals_df['time'] = signals_df['hour'] * 100 + signals_df['minute']
    signals_df['intraday'] = (signals_df['time'] >= 930) & (signals_df['time'] < 1600)
    
    print("=== INTRADAY THRESHOLD VERIFICATION ===")
    print(f"Total signals: {len(signals_df)}")
    print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
    
    # Check for after-hours signals
    after_hours = signals_df[~signals_df['intraday']]
    print(f"\nAfter-hours signals: {len(after_hours)}")
    
    if len(after_hours) > 0:
        print("\nAfter-hours signal details:")
        for _, row in after_hours.head(10).iterrows():
            print(f"  {row['ts']} - signal: {row['val']} (time: {row['time']})")
        
        # Check if all after-hours signals are 0
        non_zero_after_hours = after_hours[after_hours['val'] != 0]
        if len(non_zero_after_hours) > 0:
            print(f"\n❌ ERROR: Found {len(non_zero_after_hours)} non-zero after-hours signals!")
            print("First few:")
            for _, row in non_zero_after_hours.head(5).iterrows():
                print(f"  {row['ts']} - signal: {row['val']}")
        else:
            print("\n✅ All after-hours signals are 0 (positions closed)")
    
    # Check signal transitions at market close
    print("\n=== MARKET CLOSE TRANSITIONS ===")
    for date in signals_df['ts'].dt.date.unique()[:5]:
        day_signals = signals_df[signals_df['ts'].dt.date == date].sort_values('ts')
        
        # Find last intraday signal and first after-hours signal
        intraday_signals = day_signals[day_signals['intraday']]
        after_hours_signals = day_signals[~day_signals['intraday']]
        
        if len(intraday_signals) > 0 and len(after_hours_signals) > 0:
            last_intraday = intraday_signals.iloc[-1]
            first_after = after_hours_signals.iloc[0]
            
            print(f"\n{date}:")
            print(f"  Last intraday: {last_intraday['ts']} - signal: {last_intraday['val']}")
            print(f"  First after-hours: {first_after['ts']} - signal: {first_after['val']}")
            
            if last_intraday['val'] != 0 and first_after['val'] == 0:
                print("  ✅ Position closed at market close")
            elif last_intraday['val'] == 0:
                print("  ✅ No position to close")
            else:
                print("  ❌ Position NOT closed!")
                
except FileNotFoundError:
    print("No signal files found. Make sure to run the strategy first.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)