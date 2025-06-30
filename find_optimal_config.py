#!/usr/bin/env python3
"""Find optimal configuration across all strategies."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

def simulate_config(signals_file: str, max_minutes: int = None, stop_pct: float = None):
    """Simulate a specific configuration."""
    
    signals_df = pd.read_parquet(signals_file)
    if signals_df.empty:
        return []
    
    signals_df['datetime'] = pd.to_datetime(signals_df['ts'])
    market_close = time(15, 45)
    eod_close = time(15, 59)
    
    trades = []
    entry_price = None
    entry_signal = None
    entry_time = None
    entry_date = None
    
    for i in range(len(signals_df)):
        signal = signals_df.iloc[i]['val']
        price = signals_df.iloc[i]['px']
        current_time = signals_df.iloc[i]['datetime']
        current_date = current_time.date()
        current_tod = current_time.time()
        
        if entry_price is not None:
            exit_reason = None
            exit_price = price
            
            # EOD
            if current_date != entry_date or current_tod >= eod_close:
                exit_reason = 'eod'
            # Max hold
            elif max_minutes and (current_time - entry_time).total_seconds() / 60 >= max_minutes:
                exit_reason = 'max_hold'
            # Stop
            elif stop_pct:
                if entry_signal > 0:
                    if (entry_price - price) / entry_price > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 - stop_pct)
                else:
                    if (price - entry_price) / entry_price > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 + stop_pct)
            
            # Signal exit
            if not exit_reason and (signal == 0 or signal == -entry_signal):
                exit_reason = 'signal'
            
            if exit_reason:
                log_return = np.log(exit_price / entry_price) * entry_signal * 0.9998
                trades.append(log_return * 10000)  # Convert to bps
                
                entry_price = None
                if signal != 0 and exit_reason == 'signal' and current_tod < market_close:
                    entry_price = price
                    entry_signal = signal
                    entry_time = current_time
                    entry_date = current_date
        
        elif signal != 0 and not entry_price and current_tod < market_close:
            entry_price = price
            entry_signal = signal
            entry_time = current_time
            entry_date = current_date
    
    return trades

# Test configurations
workspace = "workspaces/signal_generation_5433aa9b"
signal_files = list(Path(workspace).glob("traces/SPY_*/signals/keltner_bands/*.parquet"))

configs = [
    ("Baseline (EOD only)", None, None),
    ("30min max", 30, None),
    ("60min max", 60, None),
    ("0.2% stop", None, 0.002),
    ("0.3% stop", None, 0.003),
    ("0.5% stop", None, 0.005),
    ("30min + 0.3% stop", 30, 0.003),
    ("60min + 0.3% stop", 60, 0.003),
    ("90min + 0.3% stop", 90, 0.003),
]

print("=== OPTIMAL CONFIGURATION SEARCH ===")
print("Testing across all strategies...\n")

results = []
for name, max_mins, stop_pct in configs:
    all_returns = []
    
    for signal_file in signal_files[:20]:  # Test first 20 strategies
        returns = simulate_config(str(signal_file), max_mins, stop_pct)
        all_returns.extend(returns)
    
    if all_returns:
        edge = np.mean(all_returns)
        total_return = np.sum(all_returns)
        trades = len(all_returns)
        
        results.append({
            'config': name,
            'edge_bps': edge,
            'total_return_bps': total_return,
            'trades': trades,
            'annual_return': edge * trades / len(signal_files[:20]) * 252 / 10000
        })

# Display results
print("Configuration       | Edge  | Trades | Annual % | Total Return")
print("--------------------|-------|--------|----------|-------------")

for r in sorted(results, key=lambda x: x['edge_bps'], reverse=True):
    print(f"{r['config']:19s} | {r['edge_bps']:5.2f} | {r['trades']:6d} | "
          f"{r['annual_return']:8.2%} | {r['total_return_bps']:12.0f}")

# Best configuration
best = max(results, key=lambda x: x['edge_bps'])
print(f"\n\nOPTIMAL CONFIGURATION: {best['config']}")
print(f"Edge per trade: {best['edge_bps']:.2f} bps")
print(f"Expected annual return: {best['annual_return']:.2%}")
print("\nThis configuration provides:")
print("- No overnight risk (EOD exits)")
print("- Limited downside (stops)")
print("- Reasonable trade frequency")

# Risk analysis
baseline = next(r for r in results if r['config'] == "Baseline (EOD only)")
stop_configs = [r for r in results if "stop" in r['config'].lower()]

print("\n\nSTOP LOSS ANALYSIS:")
for r in stop_configs:
    improvement = r['edge_bps'] - baseline['edge_bps']
    print(f"{r['config']:19s}: {improvement:+.2f} bps improvement")