#!/usr/bin/env python3
"""
Deep regime analysis for Bollinger Bands with directional filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate comprehensive indicators
# Trend indicators
prices['sma_200'] = prices['Close'].rolling(200).mean()
prices['sma_200_slope'] = prices['sma_200'].diff(10) / 10
prices['sma_50'] = prices['Close'].rolling(50).mean()
prices['sma_50_slope'] = prices['sma_50'].diff(5) / 5
prices['sma_20'] = prices['Close'].rolling(20).mean()

# Trend strength
prices['trend_strength'] = (prices['Close'] - prices['sma_200']) / prices['sma_200'] * 100

# VWAP
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()
prices['vwap_distance'] = (prices['Close'] - prices['vwap']) / prices['vwap'] * 100

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])
prices['bb_width'] = (prices['bb_upper'] - prices['bb_lower']) / prices['bb_middle'] * 100

# Volume and volatility
prices['volume_sma'] = prices['Volume'].rolling(20).mean()
prices['volume_ratio'] = prices['Volume'] / prices['volume_sma']
prices['high_low_range'] = (prices['High'] - prices['Low']) / prices['Close'] * 100
prices['volatility_20'] = prices['Close'].pct_change().rolling(20).std() * 100

# Market regimes
prices['strong_uptrend'] = (prices['sma_50'] > prices['sma_200']) & (prices['sma_50_slope'] > 0.02)
prices['strong_downtrend'] = (prices['sma_50'] < prices['sma_200']) & (prices['sma_50_slope'] < -0.02)
prices['ranging'] = (abs(prices['sma_50_slope']) < 0.01) & (prices['bb_width'] < 1.0)

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
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'entry_price': entry_price,
                
                # Position data
                'entry_bb_position': entry_row['bb_position'],
                'entry_above_vwap': entry_price > entry_row['vwap'],
                'entry_vwap_distance': entry_row['vwap_distance'],
                
                # Trend data
                'entry_sma200_slope': entry_row['sma_200_slope'],
                'entry_sma50_slope': entry_row['sma_50_slope'],
                'entry_trend_strength': entry_row['trend_strength'],
                'entry_above_sma200': entry_price > entry_row['sma_200'] if pd.notna(entry_row['sma_200']) else None,
                'entry_above_sma50': entry_price > entry_row['sma_50'] if pd.notna(entry_row['sma_50']) else None,
                
                # Regime flags
                'entry_strong_uptrend': entry_row['strong_uptrend'],
                'entry_strong_downtrend': entry_row['strong_downtrend'],
                'entry_ranging': entry_row['ranging'],
                
                # Market conditions
                'entry_volume_ratio': entry_row['volume_ratio'],
                'entry_bb_width': entry_row['bb_width'],
                'entry_volatility': entry_row['volatility_20'] if pd.notna(entry_row['volatility_20']) else None,
            })
        
        if signal != 0:
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
        else:
            entry_idx = None

trades_df = pd.DataFrame(trades)
valid_trades = trades_df.dropna(subset=['entry_sma200_slope'])

print("="*60)
print("COMPREHENSIVE REGIME ANALYSIS")
print("="*60)

# 1. Overall performance by duration
print("\n1. BASELINE PERFORMANCE BY DURATION:")
print("-" * 60)
for dur_range in [(1, 1), (2, 5), (6, 10), (11, 20)]:
    dur_trades = valid_trades[valid_trades['duration'].between(dur_range[0], dur_range[1])]
    if len(dur_trades) > 0:
        net = dur_trades['pnl_pct'].sum() - len(dur_trades) * 0.01
        win_rate = (dur_trades['pnl_pct'] > 0).mean()
        print(f"{dur_range[0]:2d}-{dur_range[1]:2d} bars: {len(dur_trades):4d} trades, {win_rate:.1%} win, {net:7.2f}% net")

# 2. Regime-specific performance
print("\n2. PERFORMANCE BY MARKET REGIME:")
print("-" * 60)

regimes = [
    ("Strong Uptrend", valid_trades['entry_strong_uptrend']),
    ("Strong Downtrend", valid_trades['entry_strong_downtrend']),
    ("Ranging Market", valid_trades['entry_ranging']),
]

for regime_name, mask in regimes:
    regime_trades = valid_trades[mask]
    if len(regime_trades) > 10:
        print(f"\n{regime_name}: {len(regime_trades)} total trades")
        
        # By duration
        for dur in [1, 2, 3, 4, 5]:
            dur_trades = regime_trades[regime_trades['duration'] == dur]
            if len(dur_trades) > 5:
                net = dur_trades['pnl_pct'].sum() - len(dur_trades) * 0.01
                win_rate = (dur_trades['pnl_pct'] > 0).mean()
                print(f"  {dur}-bar: {len(dur_trades):3d} trades, {win_rate:.1%} win, {net:6.2f}% net")

# 3. Directional filtering
print("\n3. DIRECTIONAL FILTERING ANALYSIS:")
print("-" * 60)

# Test taking only aligned trades
aligned_filters = [
    ("Longs in Uptrend Only", (valid_trades['signal_type'] == 'long') & (valid_trades['entry_sma200_slope'] > 0)),
    ("Shorts in Downtrend Only", (valid_trades['signal_type'] == 'short') & (valid_trades['entry_sma200_slope'] < 0)),
    ("Longs Below VWAP in Uptrend", (valid_trades['signal_type'] == 'long') & (~valid_trades['entry_above_vwap']) & (valid_trades['entry_sma200_slope'] > 0)),
    ("Shorts Above VWAP in Downtrend", (valid_trades['signal_type'] == 'short') & (valid_trades['entry_above_vwap']) & (valid_trades['entry_sma200_slope'] < 0)),
]

for filter_name, mask in aligned_filters:
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        print(f"\n{filter_name}:")
        print(f"Total trades: {len(filtered)}")
        
        # By duration
        for dur_range in [(1, 1), (2, 5), (6, 10)]:
            dur_trades = filtered[filtered['duration'].between(dur_range[0], dur_range[1])]
            if len(dur_trades) > 5:
                net = dur_trades['pnl_pct'].sum() - len(dur_trades) * 0.01
                win_rate = (dur_trades['pnl_pct'] > 0).mean()
                print(f"  {dur_range[0]:2d}-{dur_range[1]:2d} bars: {len(dur_trades):4d} trades, {win_rate:.1%} win, {net:7.2f}% net")

# 4. Counter-trend in ranging markets
print("\n4. COUNTER-TREND IN RANGING MARKETS:")
print("-" * 60)

ranging_trades = valid_trades[valid_trades['entry_ranging']]
if len(ranging_trades) > 10:
    print(f"Total ranging market trades: {len(ranging_trades)}")
    
    # Longs at lower band in ranging
    ranging_long_lower = ranging_trades[(ranging_trades['signal_type'] == 'long') & (ranging_trades['entry_bb_position'] < 0.2)]
    if len(ranging_long_lower) > 5:
        net = ranging_long_lower['pnl_pct'].sum() - len(ranging_long_lower) * 0.01
        print(f"\nLongs at lower band in ranging: {len(ranging_long_lower)} trades, {net:.2f}% net")
        
    # Shorts at upper band in ranging
    ranging_short_upper = ranging_trades[(ranging_trades['signal_type'] == 'short') & (ranging_trades['entry_bb_position'] > 0.8)]
    if len(ranging_short_upper) > 5:
        net = ranging_short_upper['pnl_pct'].sum() - len(ranging_short_upper) * 0.01
        print(f"Shorts at upper band in ranging: {len(ranging_short_upper)} trades, {net:.2f}% net")

# 5. Volatility-based filtering
print("\n5. VOLATILITY-BASED FILTERING:")
print("-" * 60)

# Quartiles of volatility
valid_trades['vol_quartile'] = pd.qcut(valid_trades['entry_volatility'].dropna(), q=4, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4(High)'])

for quartile in ['Q1(Low)', 'Q2', 'Q3', 'Q4(High)']:
    vol_trades = valid_trades[valid_trades['vol_quartile'] == quartile]
    if len(vol_trades) > 10:
        print(f"\n{quartile} Volatility:")
        
        # Test different durations
        for dur_range in [(1, 1), (2, 5)]:
            dur_trades = vol_trades[vol_trades['duration'].between(dur_range[0], dur_range[1])]
            if len(dur_trades) > 10:
                net = dur_trades['pnl_pct'].sum() - len(dur_trades) * 0.01
                win_rate = (dur_trades['pnl_pct'] > 0).mean()
                print(f"  {dur_range[0]:2d}-{dur_range[1]:2d} bars: {len(dur_trades):4d} trades, {win_rate:.1%} win, {net:7.2f}% net")

# 6. Ultimate filter combination
print("\n6. BEST FILTER COMBINATIONS:")
print("-" * 60)

# Test promising combinations
test_combos = [
    ("1-bar + Ranging Market", (valid_trades['duration'] == 1) & (valid_trades['entry_ranging'])),
    ("2-5 bar + Ranging + High Vol", (valid_trades['duration'].between(2, 5)) & (valid_trades['entry_ranging']) & (valid_trades['entry_volume_ratio'] > 1.5)),
    ("Aligned Direction + Low Vol", ((valid_trades['signal_type'] == 'long') & (valid_trades['entry_sma200_slope'] > 0) | (valid_trades['signal_type'] == 'short') & (valid_trades['entry_sma200_slope'] < 0)) & (valid_trades['vol_quartile'] == 'Q1(Low)')),
    ("Counter-trend + Extreme BB", ((valid_trades['signal_type'] == 'long') & (valid_trades['entry_bb_position'] < -0.1)) | ((valid_trades['signal_type'] == 'short') & (valid_trades['entry_bb_position'] > 1.1))),
]

best_results = []
for combo_name, mask in test_combos:
    combo_trades = valid_trades[mask]
    if len(combo_trades) > 20:
        net = combo_trades['pnl_pct'].sum() - len(combo_trades) * 0.01
        win_rate = (combo_trades['pnl_pct'] > 0).mean()
        avg_duration = combo_trades['duration'].mean()
        
        best_results.append((net, combo_name, len(combo_trades), win_rate, avg_duration))
        
        if net > 0:
            print(f"\n{combo_name}:")
            print(f"  Trades: {len(combo_trades)}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Net Return: {net:.2f}%")
            print(f"  Avg Duration: {avg_duration:.1f} bars")

# Show best overall
if best_results:
    best_results.sort(reverse=True)
    print("\n\nTOP 3 STRATEGIES:")
    for i, (net, name, trades, win_rate, duration) in enumerate(best_results[:3]):
        print(f"{i+1}. {name}: {net:.2f}% net ({trades} trades, {win_rate:.1%} win)")