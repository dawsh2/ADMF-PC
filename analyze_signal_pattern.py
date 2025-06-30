#!/usr/bin/env python3
"""Analyze the signal pattern to understand the poor performance."""

import pandas as pd
import numpy as np

# Load signals and market data
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
market_data = pd.read_parquet('data/SPY_5m.parquet')

print("Signal Pattern Analysis")
print("=" * 50)

# Look at a sample of signals with price context
print("\nSample signals with price context:")
print("(Bollinger mean reversion should buy at lower band, sell at upper band)")
print("-" * 70)

# Get first 20 non-zero signals
non_zero_signals = signals[signals['val'] != 0].head(20)

for _, signal in non_zero_signals.iterrows():
    idx = int(signal['idx'])
    if idx < len(market_data):
        price = market_data.iloc[idx]['close']
        
        # Calculate what Bollinger Bands would be (rough estimate)
        # For period=11, look back 11 bars
        if idx >= 11:
            window_prices = market_data.iloc[idx-10:idx+1]['close']
            sma = window_prices.mean()
            std = window_prices.std()
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            
            signal_type = "BUY" if signal['val'] == 1 else "SELL" if signal['val'] == -1 else "FLAT"
            
            print(f"Bar {idx}: {signal_type} at ${price:.2f}")
            print(f"  SMA: ${sma:.2f}, Upper: ${upper_band:.2f}, Lower: ${lower_band:.2f}")
            
            # Check if signal makes sense
            if signal['val'] == 1:  # Buy signal
                if price > lower_band + 0.1:
                    print("  ⚠️ BUY above lower band - not typical mean reversion")
            elif signal['val'] == -1:  # Sell signal  
                if price < upper_band - 0.1:
                    print("  ⚠️ SELL below upper band - not typical mean reversion")
            
            print()

# Analyze consecutive signals
print("\nConsecutive signal analysis:")
consecutive_same = 0
max_consecutive = 0
last_signal = 0

for _, signal in signals.iterrows():
    if signal['val'] != 0:
        if signal['val'] == last_signal:
            consecutive_same += 1
            max_consecutive = max(max_consecutive, consecutive_same)
        else:
            consecutive_same = 0
        last_signal = signal['val']

print(f"Max consecutive same signals: {max_consecutive}")
print("(Mean reversion should rarely have many consecutive same signals)")

# Check holding periods
print("\nHolding period analysis:")
holding_periods = []
last_entry_idx = None
last_signal_val = 0

for _, signal in signals.iterrows():
    if signal['val'] != 0 and signal['val'] != last_signal_val:
        if last_entry_idx is not None and last_signal_val != 0:
            holding_period = signal['idx'] - last_entry_idx
            holding_periods.append(holding_period)
        last_entry_idx = signal['idx']
        last_signal_val = signal['val']
    elif signal['val'] == 0 and last_signal_val != 0:
        if last_entry_idx is not None:
            holding_period = signal['idx'] - last_entry_idx
            holding_periods.append(holding_period)
        last_signal_val = 0

if holding_periods:
    holding_periods = np.array(holding_periods)
    print(f"Average holding period: {holding_periods.mean():.1f} bars")
    print(f"Median holding period: {np.median(holding_periods):.0f} bars")
    print(f"Max holding period: {holding_periods.max()} bars")
    
    # For 5-min bars, typical mean reversion should be 10-50 bars
    long_holds = (holding_periods > 100).sum()
    if long_holds > 0:
        print(f"⚠️ {long_holds} positions held > 100 bars (too long for mean reversion)")