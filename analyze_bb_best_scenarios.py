#!/usr/bin/env python3
"""
Find the best scenarios for the new Bollinger Bands strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate indicators
prices['sma_200'] = prices['Close'].rolling(200).mean()
prices['sma_200_slope'] = prices['sma_200'].diff(5) / 5
prices['sma_50'] = prices['Close'].rolling(50).mean()
prices['sma_50_slope'] = prices['sma_50'].diff(5) / 5
prices['sma_20'] = prices['Close'].rolling(20).mean()
prices['sma_20_slope'] = prices['sma_20'].diff(3) / 3

# VWAP
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])
prices['bb_width'] = (prices['bb_upper'] - prices['bb_lower']) / prices['bb_middle'] * 100

# Volume
prices['volume_sma'] = prices['Volume'].rolling(20).mean()
prices['volume_ratio'] = prices['Volume'] / prices['volume_sma']

# Extract trades with context
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
                'entry_bb_position': entry_row['bb_position'],
                'entry_bb_penetration': abs(entry_row['bb_position'] - 0.5) - 0.5,  # How far outside bands
                'entry_above_vwap': entry_price > entry_row['vwap'],
                'entry_above_sma200': entry_price > entry_row['sma_200'] if pd.notna(entry_row['sma_200']) else None,
                'entry_sma200_slope': entry_row['sma_200_slope'] if pd.notna(entry_row['sma_200_slope']) else None,
                'entry_sma50_slope': entry_row['sma_50_slope'] if pd.notna(entry_row['sma_50_slope']) else None,
                'entry_sma20_slope': entry_row['sma_20_slope'] if pd.notna(entry_row['sma_20_slope']) else None,
                'entry_bb_width': entry_row['bb_width'] if pd.notna(entry_row['bb_width']) else None,
                'entry_volume_ratio': entry_row['volume_ratio'] if pd.notna(entry_row['volume_ratio']) else None,
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
print("FINDING PROFITABLE SCENARIOS")
print("="*60)

# Test various filter combinations
filters = {
    "1-bar trades only": valid_trades['duration'] == 1,
    "2-5 bar trades": valid_trades['duration'].between(2, 5),
    "High volume (>1.5x avg)": valid_trades['entry_volume_ratio'] > 1.5,
    "Low volume (<0.7x avg)": valid_trades['entry_volume_ratio'] < 0.7,
    "Wide BB (>1%)": valid_trades['entry_bb_width'] > 1.0,
    "Narrow BB (<0.5%)": valid_trades['entry_bb_width'] < 0.5,
    "Deep penetration (>5% outside)": valid_trades['entry_bb_penetration'] > 0.05,
    "Shallow penetration (<2% outside)": valid_trades['entry_bb_penetration'] < 0.02,
    "SMA20 rising fast": valid_trades['entry_sma20_slope'] > 0.05,
    "SMA20 falling fast": valid_trades['entry_sma20_slope'] < -0.05,
}

print("\n1. SINGLE FILTER ANALYSIS:")
print("-" * 60)

single_filter_results = []
for name, mask in filters.items():
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        gross = filtered['pnl_pct'].sum()
        net = gross - len(filtered) * 0.01
        win_rate = (filtered['pnl_pct'] > 0).mean()
        
        single_filter_results.append({
            'filter': name,
            'trades': len(filtered),
            'win_rate': win_rate,
            'gross': gross,
            'net': net,
            'net_per_trade': net / len(filtered)
        })
        
        if net > 0:  # Only show profitable filters
            print(f"\n{name}:")
            print(f"  Trades: {len(filtered)}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Net Return: {net:.2f}%")
            print(f"  Per Trade: {net/len(filtered):.3f}%")

# Test combinations of profitable filters
print("\n\n2. COMBINED FILTER ANALYSIS:")
print("-" * 60)

# Focus on 1-bar trades with additional filters
base_filter = valid_trades['duration'] == 1

additional_filters = [
    ("+ Below VWAP", ~valid_trades['entry_above_vwap']),
    ("+ Above VWAP", valid_trades['entry_above_vwap']),
    ("+ High Volume", valid_trades['entry_volume_ratio'] > 1.5),
    ("+ Wide BB", valid_trades['entry_bb_width'] > 1.0),
    ("+ Deep Penetration", valid_trades['entry_bb_penetration'] > 0.05),
    ("+ SMA200 Rising", valid_trades['entry_sma200_slope'] > 0),
    ("+ SMA200 Falling", valid_trades['entry_sma200_slope'] < 0),
]

for name, additional_mask in additional_filters:
    combined = valid_trades[base_filter & additional_mask]
    if len(combined) > 10:
        gross = combined['pnl_pct'].sum()
        net = gross - len(combined) * 0.01
        win_rate = (combined['pnl_pct'] > 0).mean()
        
        if net > 5:  # Show very profitable combinations
            print(f"\n1-bar trades {name}:")
            print(f"  Trades: {len(combined)}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Net Return: {net:.2f}%")
            print(f"  Per Trade: {net/len(combined):.3f}%")

# Analyze by signal type
print("\n\n3. SIGNAL TYPE ANALYSIS:")
print("-" * 60)

for signal_type in ['long', 'short']:
    type_trades = valid_trades[valid_trades['signal_type'] == signal_type]
    
    # Best scenarios for this signal type
    scenarios = [
        ("All", type_trades),
        ("1-bar only", type_trades[type_trades['duration'] == 1]),
        ("Below VWAP", type_trades[~type_trades['entry_above_vwap']]),
        ("High Volume", type_trades[type_trades['entry_volume_ratio'] > 1.5]),
        ("1-bar + Below VWAP", type_trades[(type_trades['duration'] == 1) & (~type_trades['entry_above_vwap'])]),
    ]
    
    print(f"\n{signal_type.upper()} SIGNALS:")
    for scenario_name, scenario_trades in scenarios:
        if len(scenario_trades) > 0:
            net = scenario_trades['pnl_pct'].sum() - len(scenario_trades) * 0.01
            win_rate = (scenario_trades['pnl_pct'] > 0).mean()
            print(f"  {scenario_name}: {len(scenario_trades)} trades, {win_rate:.1%} win, {net:.2f}% net")

# Find the absolute best combination
print("\n\n4. OPTIMAL STRATEGY:")
print("-" * 60)

# Test all reasonable combinations
best_net = -100
best_combo = None

test_combos = [
    ("1-bar + Below VWAP", (valid_trades['duration'] == 1) & (~valid_trades['entry_above_vwap'])),
    ("1-bar + High Volume", (valid_trades['duration'] == 1) & (valid_trades['entry_volume_ratio'] > 1.5)),
    ("1-bar + Wide BB", (valid_trades['duration'] == 1) & (valid_trades['entry_bb_width'] > 1.0)),
    ("1-bar + Below VWAP + High Vol", (valid_trades['duration'] == 1) & (~valid_trades['entry_above_vwap']) & (valid_trades['entry_volume_ratio'] > 1.5)),
]

for name, mask in test_combos:
    filtered = valid_trades[mask]
    if len(filtered) > 20:  # Minimum trades for reliability
        net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
        if net > best_net:
            best_net = net
            best_combo = (name, filtered)

if best_combo:
    name, best_trades = best_combo
    print(f"\nBEST COMBINATION: {name}")
    print(f"Trades: {len(best_trades)}")
    print(f"Win Rate: {(best_trades['pnl_pct'] > 0).mean():.1%}")
    print(f"Net Return: {best_net:.2f}%")
    print(f"Average per trade: {best_net/len(best_trades):.3f}%")
    
    # Breakdown by signal type
    for sig_type in ['long', 'short']:
        sig_trades = best_trades[best_trades['signal_type'] == sig_type]
        if len(sig_trades) > 0:
            sig_net = sig_trades['pnl_pct'].sum() - len(sig_trades) * 0.01
            print(f"\n  {sig_type.upper()}: {len(sig_trades)} trades, {sig_net:.2f}% net")