#!/usr/bin/env python3
"""Analyze RSI divergence detection to see what's happening."""

import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_parquet('data/SPY_1m.parquet')

# Filter to test period (last 20% of data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
train_size = int(len(df) * 0.8)
df = df.iloc[train_size:]
print(f"Analyzing TEST SET: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

# Calculate RSI manually
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = np.nan
    rsi[period] = 100. - 100. / (1. + rs)
    
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

# Calculate RSI
df['rsi'] = calculate_rsi(df['close'].values)

# Find local extremes
def find_local_extremes(prices, window=5):
    lows = []
    highs = []
    
    for i in range(window, len(prices) - window):
        # Local low
        if all(prices[i] <= prices[j] for j in range(i-window, i+window+1) if j != i):
            lows.append(i)
        # Local high  
        if all(prices[i] >= prices[j] for j in range(i-window, i+window+1) if j != i):
            highs.append(i)
    
    return lows, highs

# Find extremes
lows, highs = find_local_extremes(df['close'].values, window=3)
print(f"Found {len(lows)} local lows and {len(highs)} local highs")

# Check for divergences
divergences = []
lookback = 30
min_bars = 3
rsi_threshold = 3.0
price_threshold = 0.0005

# Check bullish divergences
for i in range(1, len(lows)):
    curr_idx = lows[i]
    
    # Look back for previous lows
    for j in range(i-1, max(0, i-10), -1):
        prev_idx = lows[j]
        
        if curr_idx - prev_idx > lookback:
            break
        if curr_idx - prev_idx < min_bars:
            continue
            
        # Check price and RSI
        curr_price = df.iloc[curr_idx]['close']
        prev_price = df.iloc[prev_idx]['close']
        curr_rsi = df.iloc[curr_idx]['rsi']
        prev_rsi = df.iloc[prev_idx]['rsi']
        
        price_change = (curr_price - prev_price) / prev_price
        
        # Bullish divergence: price lower, RSI higher
        if price_change < -price_threshold and curr_rsi > prev_rsi + rsi_threshold:
            divergences.append({
                'type': 'bullish',
                'prev_idx': prev_idx,
                'curr_idx': curr_idx,
                'prev_price': prev_price,
                'curr_price': curr_price,
                'prev_rsi': prev_rsi,
                'curr_rsi': curr_rsi,
                'price_change': price_change * 100,
                'rsi_diff': curr_rsi - prev_rsi,
                'timestamp': df.iloc[curr_idx]['timestamp']
            })
            break

# Check bearish divergences
for i in range(1, len(highs)):
    curr_idx = highs[i]
    
    # Look back for previous highs
    for j in range(i-1, max(0, i-10), -1):
        prev_idx = highs[j]
        
        if curr_idx - prev_idx > lookback:
            break
        if curr_idx - prev_idx < min_bars:
            continue
            
        # Check price and RSI
        curr_price = df.iloc[curr_idx]['close']
        prev_price = df.iloc[prev_idx]['close']
        curr_rsi = df.iloc[curr_idx]['rsi']
        prev_rsi = df.iloc[prev_idx]['rsi']
        
        price_change = (curr_price - prev_price) / prev_price
        
        # Bearish divergence: price higher, RSI lower
        if price_change > price_threshold and curr_rsi < prev_rsi - rsi_threshold:
            divergences.append({
                'type': 'bearish',
                'prev_idx': prev_idx,
                'curr_idx': curr_idx,
                'prev_price': prev_price,
                'curr_price': curr_price,
                'prev_rsi': prev_rsi,
                'curr_rsi': curr_rsi,
                'price_change': price_change * 100,
                'rsi_diff': curr_rsi - prev_rsi,
                'timestamp': df.iloc[curr_idx]['timestamp']
            })
            break

print(f"\nFound {len(divergences)} divergences")
if divergences:
    for div in divergences[:10]:
        print(f"{div['type']} divergence at {div['timestamp']}: "
              f"price change {div['price_change']:.2f}%, RSI diff {div['rsi_diff']:.1f}")

# Summary statistics
if divergences:
    bullish = [d for d in divergences if d['type'] == 'bullish']
    bearish = [d for d in divergences if d['type'] == 'bearish']
    print(f"\nBullish divergences: {len(bullish)}")
    print(f"Bearish divergences: {len(bearish)}")
    
    # Show timing
    div_df = pd.DataFrame(divergences)
    div_df['timestamp'] = pd.to_datetime(div_df['timestamp'])
    div_df['date'] = div_df['timestamp'].dt.date
    
    print("\nDivergences by date:")
    print(div_df.groupby(['date', 'type']).size().unstack(fill_value=0))