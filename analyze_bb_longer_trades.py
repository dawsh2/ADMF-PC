#!/usr/bin/env python3
"""
Analyze longer-duration Bollinger Bands trades with filters
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals (using the original workspace, not the buffer one)
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_946b8943/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate indicators
prices['sma_200'] = prices['Close'].rolling(200).mean()
prices['sma_200_slope'] = prices['sma_200'].diff(10) / 10  # 10-bar slope for stability
prices['sma_50'] = prices['Close'].rolling(50).mean()

# VWAP
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# Volume metrics
prices['volume_sma'] = prices['Volume'].rolling(20).mean()
prices['volume_ratio'] = prices['Volume'] / prices['volume_sma']

# ATR for volatility
prices['tr'] = prices[['High', 'Low']].max(axis=1) - prices[['High', 'Low']].min(axis=1)
prices['atr'] = prices['tr'].rolling(14).mean()
prices['atr_pct'] = prices['atr'] / prices['Close'] * 100

# Extract trades with full context
trades = []
entry_idx = None
entry_signal = None

for _, row in signals.iterrows():
    signal = row['val']
    bar_idx = row['idx']
    
    if entry_idx is None and signal != 0:
        if bar_idx < len(prices):
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
            
    elif entry_idx is not None and (signal == 0 or signal != entry_signal):
        # Exit
        if entry_signal > 0:
            pnl_pct = (row['px'] - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - row['px']) / entry_price * 100
            
        if bar_idx < len(prices) and entry_idx < len(prices):
            entry_row = prices.iloc[entry_idx]
            
            # Calculate how far outside bands
            if entry_row['bb_position'] > 1:
                penetration = (entry_row['Close'] - entry_row['bb_upper']) / entry_row['bb_upper'] * 100
            elif entry_row['bb_position'] < 0:
                penetration = (entry_row['bb_lower'] - entry_row['Close']) / entry_row['bb_lower'] * 100
            else:
                penetration = 0
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'entry_price': entry_price,
                'entry_bb_position': entry_row['bb_position'],
                'entry_penetration_pct': penetration,
                'entry_above_vwap': entry_price > entry_row['vwap'],
                'entry_above_sma200': entry_price > entry_row['sma_200'] if pd.notna(entry_row['sma_200']) else None,
                'entry_sma200_slope': entry_row['sma_200_slope'] if pd.notna(entry_row['sma_200_slope']) else None,
                'entry_volume_ratio': entry_row['volume_ratio'],
                'entry_atr_pct': entry_row['atr_pct'] if pd.notna(entry_row['atr_pct']) else None,
            })
        
        # Check for re-entry
        if signal != 0:
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
        else:
            entry_idx = None

trades_df = pd.DataFrame(trades)
valid_trades = trades_df.dropna(subset=['entry_sma200_slope', 'entry_above_vwap'])

print("="*60)
print("LONGER DURATION TRADE ANALYSIS")
print("="*60)

# Focus on trades 2-10 bars (not 1-bar scalps)
longer_trades = valid_trades[valid_trades['duration'].between(2, 10)]
print(f"\nTotal trades 2-10 bars: {len(longer_trades)}")
print(f"Overall performance: {longer_trades['pnl_pct'].sum() - len(longer_trades) * 0.01:.2f}% net")

# 1. Analyze by volume
print("\n1. VOLUME ANALYSIS FOR 2-10 BAR TRADES:")
print("-" * 60)

volume_buckets = [(0, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 10)]
for low, high in volume_buckets:
    vol_trades = longer_trades[longer_trades['entry_volume_ratio'].between(low, high)]
    if len(vol_trades) > 10:
        net = vol_trades['pnl_pct'].sum() - len(vol_trades) * 0.01
        win_rate = (vol_trades['pnl_pct'] > 0).mean()
        print(f"Volume {low:.1f}-{high:.1f}x: {len(vol_trades):4d} trades, {win_rate:.1%} win, {net:6.2f}% net")

# 2. Long/Short performance by regime
print("\n2. LONG/SHORT PERFORMANCE BY REGIME:")
print("-" * 60)

for signal_type in ['long', 'short']:
    print(f"\n{signal_type.upper()} SIGNALS (2-10 bar trades):")
    type_trades = longer_trades[longer_trades['signal_type'] == signal_type]
    
    # Analyze by SMA slope and VWAP position
    regimes = [
        ("Uptrend + Above VWAP", (type_trades['entry_sma200_slope'] > 0) & (type_trades['entry_above_vwap'])),
        ("Uptrend + Below VWAP", (type_trades['entry_sma200_slope'] > 0) & (~type_trades['entry_above_vwap'])),
        ("Downtrend + Above VWAP", (type_trades['entry_sma200_slope'] < 0) & (type_trades['entry_above_vwap'])),
        ("Downtrend + Below VWAP", (type_trades['entry_sma200_slope'] < 0) & (~type_trades['entry_above_vwap'])),
    ]
    
    for regime_name, mask in regimes:
        regime_trades = type_trades[mask]
        if len(regime_trades) > 5:
            net = regime_trades['pnl_pct'].sum() - len(regime_trades) * 0.01
            win_rate = (regime_trades['pnl_pct'] > 0).mean()
            avg_pnl = regime_trades['pnl_pct'].mean()
            print(f"  {regime_name:25s}: {len(regime_trades):3d} trades, {win_rate:5.1%} win, {net:6.2f}% net, {avg_pnl:6.3f}% avg")

# 3. Deep penetration analysis
print("\n3. PENETRATION DEPTH ANALYSIS (2-10 bar trades):")
print("-" * 60)

penetration_buckets = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 10)]
for low, high in penetration_buckets:
    pen_trades = longer_trades[longer_trades['entry_penetration_pct'].between(low, high)]
    if len(pen_trades) > 10:
        net = pen_trades['pnl_pct'].sum() - len(pen_trades) * 0.01
        win_rate = (pen_trades['pnl_pct'] > 0).mean()
        print(f"Penetration {low:.1f}-{high:.1f}%: {len(pen_trades):4d} trades, {win_rate:.1%} win, {net:6.2f}% net")

# 4. Find profitable combinations
print("\n4. PROFITABLE FILTER COMBINATIONS (2-10 bar trades):")
print("-" * 60)

# Test various filters
filter_combos = [
    ("High Volume (>1.5x)", longer_trades['entry_volume_ratio'] > 1.5),
    ("Deep Penetration (>0.5%)", longer_trades['entry_penetration_pct'] > 0.5),
    ("High Vol + Deep Pen", (longer_trades['entry_volume_ratio'] > 1.5) & (longer_trades['entry_penetration_pct'] > 0.5)),
    ("Longs Below VWAP", (longer_trades['signal_type'] == 'long') & (~longer_trades['entry_above_vwap'])),
    ("Shorts Above VWAP", (longer_trades['signal_type'] == 'short') & (longer_trades['entry_above_vwap'])),
    ("High Vol + Longs Below VWAP", (longer_trades['entry_volume_ratio'] > 1.5) & (longer_trades['signal_type'] == 'long') & (~longer_trades['entry_above_vwap'])),
]

profitable_combos = []
for name, mask in filter_combos:
    filtered = longer_trades[mask]
    if len(filtered) > 10:
        net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
        win_rate = (filtered['pnl_pct'] > 0).mean()
        avg_duration = filtered['duration'].mean()
        
        if net > 0:  # Only show profitable
            profitable_combos.append((net, name, filtered))
            print(f"\n{name}:")
            print(f"  Trades: {len(filtered)}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Net Return: {net:.2f}%")
            print(f"  Avg Duration: {avg_duration:.1f} bars")

# 5. Best overall strategy
print("\n5. OPTIMAL LONGER-DURATION STRATEGY:")
print("-" * 60)

if profitable_combos:
    best_net, best_name, best_trades = max(profitable_combos, key=lambda x: x[0])
    
    print(f"\nBest filter: {best_name}")
    print(f"Total trades: {len(best_trades)}")
    print(f"Net return: {best_net:.2f}%")
    print(f"Win rate: {(best_trades['pnl_pct'] > 0).mean():.1%}")
    print(f"Average duration: {best_trades['duration'].mean():.1f} bars")
    
    # Show duration breakdown
    print("\nDuration breakdown:")
    for d in range(2, 11):
        d_trades = best_trades[best_trades['duration'] == d]
        if len(d_trades) > 0:
            d_net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            print(f"  {d} bars: {len(d_trades)} trades, {d_net:.2f}% net")
else:
    print("\nNo profitable combinations found for 2-10 bar trades!")

# 6. Alternative: What about slightly longer holds?
print("\n6. CHECKING 11-20 BAR TRADES:")
print("-" * 60)

medium_trades = valid_trades[valid_trades['duration'].between(11, 20)]
print(f"Total 11-20 bar trades: {len(medium_trades)}")

if len(medium_trades) > 10:
    # Try with filters
    test_filters = [
        ("All 11-20 bar", medium_trades),
        ("High Volume", medium_trades[medium_trades['entry_volume_ratio'] > 1.5]),
        ("Deep Penetration", medium_trades[medium_trades['entry_penetration_pct'] > 0.5]),
    ]
    
    for name, filtered in test_filters:
        if len(filtered) > 5:
            net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
            print(f"{name}: {len(filtered)} trades, {net:.2f}% net")