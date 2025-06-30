#!/usr/bin/env python3
"""
Analyze Bollinger Bands with confirmation and divergence strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load price data first to implement new strategies
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate indicators
# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# How far outside bands (percentage)
prices['bb_penetration_lower'] = np.where(prices['Close'] < prices['bb_lower'], 
                                          (prices['bb_lower'] - prices['Close']) / prices['bb_lower'] * 100, 0)
prices['bb_penetration_upper'] = np.where(prices['Close'] > prices['bb_upper'], 
                                          (prices['Close'] - prices['bb_upper']) / prices['bb_upper'] * 100, 0)

# RSI for divergence
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

prices['rsi'] = calculate_rsi(prices['Close'])

# Candle patterns
prices['is_green'] = prices['Close'] > prices['Open']
prices['prev_high'] = prices['High'].shift(1)
prices['closes_above_prev_high'] = prices['Close'] > prices['prev_high']

# Volume
prices['volume_sma'] = prices['Volume'].rolling(20).mean()
prices['volume_ratio'] = prices['Volume'] / prices['volume_sma']

# SMA for trend context
prices['sma_200'] = prices['Close'].rolling(200).mean()
prices['sma_200_slope'] = prices['sma_200'].diff(10) / 10

print("="*60)
print("CONFIRMATION-BASED BOLLINGER BAND STRATEGIES")
print("="*60)

# Strategy 1: Mean Reversion with Confirmation
print("\n1. MEAN REVERSION WITH CONFIRMATION STRATEGY:")
print("-" * 60)

# Find potential long entries
long_signals = []
short_signals = []

for i in range(30, len(prices) - 50):  # Leave room for analysis
    
    # LONG SETUP: Price >1% below lower band, then green candle closing above previous high
    if prices.iloc[i]['bb_penetration_lower'] > 1.0:  # Start with 1% instead of 3%
        # Look for confirmation in next 5 bars
        for j in range(i + 1, min(i + 6, len(prices))):
            if (prices.iloc[j]['is_green'] and 
                prices.iloc[j]['closes_above_prev_high'] and
                prices.iloc[j]['Close'] > prices.iloc[i]['Low']):  # Must be above the extreme low
                
                # Entry signal
                entry_idx = j
                entry_price = prices.iloc[j]['Close']
                stop_loss = prices.iloc[i]['Low'] * 0.998  # Just below extreme low
                
                # Find exit (middle band, upper band, or stop)
                for k in range(entry_idx + 1, min(entry_idx + 50, len(prices))):
                    if (prices.iloc[k]['Close'] >= prices.iloc[k]['bb_middle'] or  # Target hit
                        prices.iloc[k]['Low'] <= stop_loss):  # Stop hit
                        
                        exit_idx = k
                        exit_price = prices.iloc[k]['Close']
                        
                        if prices.iloc[k]['Low'] <= stop_loss:
                            exit_price = stop_loss  # Stopped out
                        
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        
                        long_signals.append({
                            'entry_idx': entry_idx,
                            'exit_idx': exit_idx,
                            'duration': exit_idx - entry_idx,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_pct,
                            'penetration': prices.iloc[i]['bb_penetration_lower'],
                            'volume_ratio': prices.iloc[entry_idx]['volume_ratio'],
                            'sma_slope': prices.iloc[entry_idx]['sma_200_slope'] if pd.notna(prices.iloc[entry_idx]['sma_200_slope']) else 0
                        })
                        break
                break

# Convert to DataFrame
long_df = pd.DataFrame(long_signals) if long_signals else pd.DataFrame()

if len(long_df) > 0:
    print(f"Total long trades: {len(long_df)}")
    print(f"Win rate: {(long_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL: {long_df['pnl_pct'].mean():.3f}%")
    print(f"Net return: {long_df['pnl_pct'].sum() - len(long_df) * 0.01:.2f}%")
    
    # Duration analysis
    print("\nBy duration:")
    for d in range(1, 11):
        d_trades = long_df[long_df['duration'] == d]
        if len(d_trades) > 0:
            net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            print(f"  {d} bars: {len(d_trades)} trades, {net:.2f}% net")
    
    # By penetration depth
    print("\nBy penetration depth:")
    for low, high in [(1, 2), (2, 3), (3, 5), (5, 10)]:
        pen_trades = long_df[long_df['penetration'].between(low, high)]
        if len(pen_trades) > 0:
            net = pen_trades['pnl_pct'].sum() - len(pen_trades) * 0.01
            win_rate = (pen_trades['pnl_pct'] > 0).mean()
            print(f"  {low}-{high}%: {len(pen_trades)} trades, {win_rate:.1%} win, {net:.2f}% net")

# Strategy 2: RSI Divergence
print("\n\n2. BOLLINGER + RSI DIVERGENCE STRATEGY:")
print("-" * 60)

# Find divergences
divergence_signals = []

# Look for bullish divergence
for i in range(50, len(prices) - 50):
    # Price below lower band
    if prices.iloc[i]['Close'] < prices.iloc[i]['bb_lower']:
        # Look back for previous low below band
        for j in range(max(i - 20, 50), i):
            if prices.iloc[j]['Close'] < prices.iloc[j]['bb_lower']:
                # Check if current low is lower than previous (price wise)
                # But RSI is higher (bullish divergence)
                if (prices.iloc[i]['Low'] < prices.iloc[j]['Low'] and 
                    prices.iloc[i]['rsi'] > prices.iloc[j]['rsi'] + 5):  # RSI higher by at least 5 points
                    
                    # Wait for confirmation: price closes back inside bands
                    for k in range(i + 1, min(i + 10, len(prices))):
                        if prices.iloc[k]['Close'] > prices.iloc[k]['bb_lower']:
                            entry_idx = k
                            entry_price = prices.iloc[k]['Close']
                            
                            # Exit at middle or upper band
                            for m in range(entry_idx + 1, min(entry_idx + 50, len(prices))):
                                if prices.iloc[m]['Close'] >= prices.iloc[m]['bb_middle']:
                                    exit_idx = m
                                    exit_price = prices.iloc[m]['Close']
                                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                                    
                                    divergence_signals.append({
                                        'entry_idx': entry_idx,
                                        'exit_idx': exit_idx,
                                        'duration': exit_idx - entry_idx,
                                        'pnl_pct': pnl_pct,
                                        'rsi_at_entry': prices.iloc[entry_idx]['rsi'],
                                        'price_low_diff': (prices.iloc[i]['Low'] - prices.iloc[j]['Low']) / prices.iloc[j]['Low'] * 100,
                                        'rsi_diff': prices.iloc[i]['rsi'] - prices.iloc[j]['rsi']
                                    })
                                    break
                            break
                    break

div_df = pd.DataFrame(divergence_signals) if divergence_signals else pd.DataFrame()

if len(div_df) > 0:
    print(f"Total divergence trades: {len(div_df)}")
    print(f"Win rate: {(div_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL: {div_df['pnl_pct'].mean():.3f}%")
    print(f"Net return: {div_df['pnl_pct'].sum() - len(div_df) * 0.01:.2f}%")
    
    # Duration analysis
    print("\nBy duration:")
    duration_summary = []
    for d_range in [(1, 5), (6, 10), (11, 20), (21, 50)]:
        d_trades = div_df[div_df['duration'].between(d_range[0], d_range[1])]
        if len(d_trades) > 0:
            net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            win_rate = (d_trades['pnl_pct'] > 0).mean()
            avg_pnl = d_trades['pnl_pct'].mean()
            print(f"  {d_range[0]}-{d_range[1]} bars: {len(d_trades)} trades, {win_rate:.1%} win, "
                  f"{avg_pnl:.3f}% avg, {net:.2f}% net")

# Strategy 3: Extreme penetration with volume
print("\n\n3. EXTREME PENETRATION + VOLUME STRATEGY:")
print("-" * 60)

extreme_signals = []

for i in range(30, len(prices) - 50):
    # Look for extreme penetration with high volume
    if (prices.iloc[i]['bb_penetration_lower'] > 2.0 and  # 2%+ below lower band
        prices.iloc[i]['volume_ratio'] > 1.5):  # High volume
        
        # Enter on next bar if it's not making new lows
        if i + 1 < len(prices) and prices.iloc[i + 1]['Low'] >= prices.iloc[i]['Low']:
            entry_idx = i + 1
            entry_price = prices.iloc[entry_idx]['Close']
            
            # Exit strategy: Hold for fixed duration or hit target
            target_price = entry_price * 1.01  # 1% target
            stop_price = prices.iloc[i]['Low'] * 0.995  # Stop below extreme
            
            for j in range(entry_idx + 1, min(entry_idx + 20, len(prices))):
                if (prices.iloc[j]['High'] >= target_price or 
                    prices.iloc[j]['Low'] <= stop_price or
                    j == entry_idx + 10):  # Max 10 bars
                    
                    if prices.iloc[j]['High'] >= target_price:
                        exit_price = target_price
                    elif prices.iloc[j]['Low'] <= stop_price:
                        exit_price = stop_price
                    else:
                        exit_price = prices.iloc[j]['Close']
                    
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    
                    extreme_signals.append({
                        'entry_idx': entry_idx,
                        'exit_idx': j,
                        'duration': j - entry_idx,
                        'pnl_pct': pnl_pct,
                        'penetration': prices.iloc[i]['bb_penetration_lower'],
                        'volume_ratio': prices.iloc[i]['volume_ratio']
                    })
                    break

extreme_df = pd.DataFrame(extreme_signals) if extreme_signals else pd.DataFrame()

if len(extreme_df) > 0:
    print(f"Total extreme penetration trades: {len(extreme_df)}")
    print(f"Win rate: {(extreme_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL: {extreme_df['pnl_pct'].mean():.3f}%")
    print(f"Net return: {extreme_df['pnl_pct'].sum() - len(extreme_df) * 0.01:.2f}%")
    
    # Best combinations
    print("\nBest scenarios:")
    # Very extreme penetration
    very_extreme = extreme_df[extreme_df['penetration'] > 3.0]
    if len(very_extreme) > 0:
        net = very_extreme['pnl_pct'].sum() - len(very_extreme) * 0.01
        print(f"  Penetration >3%: {len(very_extreme)} trades, {net:.2f}% net")
    
    # High volume
    high_vol = extreme_df[extreme_df['volume_ratio'] > 2.0]
    if len(high_vol) > 0:
        net = high_vol['pnl_pct'].sum() - len(high_vol) * 0.01
        print(f"  Volume >2x: {len(high_vol)} trades, {net:.2f}% net")

# Summary
print("\n\nSUMMARY OF CONFIRMATION STRATEGIES:")
print("="*60)

all_strategies = [
    ("Mean Reversion with Confirmation", long_df if len(long_df) > 0 else pd.DataFrame()),
    ("RSI Divergence", div_df if len(div_df) > 0 else pd.DataFrame()),
    ("Extreme Penetration + Volume", extreme_df if len(extreme_df) > 0 else pd.DataFrame())
]

for name, df in all_strategies:
    if len(df) > 0:
        net = df['pnl_pct'].sum() - len(df) * 0.01
        win_rate = (df['pnl_pct'] > 0).mean()
        avg_duration = df['duration'].mean()
        print(f"\n{name}:")
        print(f"  Trades: {len(df)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Net Return: {net:.2f}%")
        print(f"  Avg Duration: {avg_duration:.1f} bars")