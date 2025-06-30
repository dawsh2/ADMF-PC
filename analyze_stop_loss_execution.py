#!/usr/bin/env python3
"""
Analyze why we're hitting stop losses 45% of the time vs notebook's 20.7%
"""
import pandas as pd
import numpy as np

# Read the data
positions_open = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/portfolio/positions_open/positions_open.parquet')
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/portfolio/positions_close/positions_close.parquet')
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print("=== Stop Loss Execution Analysis ===\n")

# First, let's check our stop loss percentage
stop_loss_pct = 0.00075  # 0.075% from config

# Analyze trades that hit stop losses
stop_loss_trades = []
for i in range(min(len(positions_open), len(positions_close))):
    close_meta = positions_close.iloc[i]['metadata']
    if isinstance(close_meta, dict):
        exit_type = close_meta.get('metadata', {}).get('exit_type', 'unknown')
        if exit_type == 'stop_loss':
            stop_loss_trades.append(i)

print(f"Total trades: {len(positions_close)}")
print(f"Stop losses hit: {len(stop_loss_trades)} ({len(stop_loss_trades)/len(positions_close)*100:.1f}%)")
print(f"Expected from notebook: ~20.7%\n")

# Examine some stop loss trades in detail
print("=== Stop Loss Trade Examples ===")
for idx in stop_loss_trades[:5]:
    open_data = positions_open.iloc[idx]
    close_data = positions_close.iloc[idx]
    
    open_meta = open_data['metadata']
    close_meta = close_data['metadata']
    
    if isinstance(open_meta, dict) and isinstance(close_meta, dict):
        qty = open_meta.get('quantity', 0)
        entry_price = open_meta.get('entry_price', 0)
        exit_price = close_meta.get('exit_price', 0)
        
        print(f"\nTrade {idx + 1}:")
        print(f"  Type: {'LONG' if qty > 0 else 'SHORT'}")
        print(f"  Entry bar: {open_data['idx']}")
        print(f"  Exit bar: {close_data['idx']}")
        print(f"  Bars held: {close_data['idx'] - open_data['idx']}")
        print(f"  Entry price: ${entry_price:.2f}")
        print(f"  Exit price: ${exit_price:.2f}")
        
        # Calculate actual vs expected stop
        if qty > 0:  # Long
            expected_stop = entry_price * (1 - stop_loss_pct)
            actual_loss_pct = (entry_price - exit_price) / entry_price * 100
        else:  # Short
            expected_stop = entry_price * (1 + stop_loss_pct)
            actual_loss_pct = (exit_price - entry_price) / entry_price * 100
        
        print(f"  Expected stop: ${expected_stop:.2f}")
        print(f"  Actual loss %: {actual_loss_pct:.3f}%")
        
        # Get OHLC data for entry and exit bars
        entry_bar_data = signals[signals['idx'] == open_data['idx']]
        exit_bar_data = signals[signals['idx'] == close_data['idx']]
        
        if len(entry_bar_data) > 0:
            entry_meta = entry_bar_data.iloc[0]['metadata']
            if isinstance(entry_meta, dict):
                print(f"  Entry bar OHLC: O=${entry_meta.get('open', 0):.2f}, "
                      f"H=${entry_meta.get('high', 0):.2f}, "
                      f"L=${entry_meta.get('low', 0):.2f}, "
                      f"C=${entry_meta.get('close', 0):.2f}")
        
        if len(exit_bar_data) > 0:
            exit_meta = exit_bar_data.iloc[0]['metadata']
            if isinstance(exit_meta, dict):
                print(f"  Exit bar OHLC: O=${exit_meta.get('open', 0):.2f}, "
                      f"H=${exit_meta.get('high', 0):.2f}, "
                      f"L=${exit_meta.get('low', 0):.2f}, "
                      f"C=${exit_meta.get('close', 0):.2f}")
                
                # Check if stop was hit within the bar's range
                if qty > 0:  # Long - check if low went below stop
                    if exit_meta.get('low', 0) <= expected_stop:
                        print(f"  ✓ Stop correctly triggered (low ${exit_meta.get('low', 0):.2f} <= stop ${expected_stop:.2f})")
                    else:
                        print(f"  ❌ Stop triggered but low ${exit_meta.get('low', 0):.2f} > stop ${expected_stop:.2f}")
                else:  # Short - check if high went above stop
                    if exit_meta.get('high', 0) >= expected_stop:
                        print(f"  ✓ Stop correctly triggered (high ${exit_meta.get('high', 0):.2f} >= stop ${expected_stop:.2f})")
                    else:
                        print(f"  ❌ Stop triggered but high ${exit_meta.get('high', 0):.2f} < stop ${expected_stop:.2f}")

print("\n=== Stop Loss Timing Analysis ===")

# Analyze when stops are being hit
bars_to_stop = []
for idx in stop_loss_trades:
    open_bar = positions_open.iloc[idx]['idx']
    close_bar = positions_close.iloc[idx]['idx']
    bars_to_stop.append(close_bar - open_bar)

if bars_to_stop:
    print(f"Average bars until stop hit: {np.mean(bars_to_stop):.1f}")
    print(f"Median bars until stop hit: {np.median(bars_to_stop):.0f}")
    print(f"Stops hit on next bar: {sum(1 for b in bars_to_stop if b == 1)} ({sum(1 for b in bars_to_stop if b == 1)/len(bars_to_stop)*100:.1f}%)")
    print(f"Stops hit within 5 bars: {sum(1 for b in bars_to_stop if b <= 5)} ({sum(1 for b in bars_to_stop if b <= 5)/len(bars_to_stop)*100:.1f}%)")

print("\n=== Hypothesis ===")
print("Possible reasons for higher stop loss rate:")
print("1. Entry execution price difference (notebook might enter at better prices)")
print("2. Stop calculation method (notebook might use different reference price)")
print("3. Intrabar execution (notebook might check stops differently)")
print("4. Market volatility during our test period vs notebook's period")

# Check volatility
print("\n=== Volatility Check ===")
# Calculate 5-min returns for volatility estimate
returns = []
for i in range(1, len(signals)):
    curr_meta = signals.iloc[i]['metadata']
    prev_meta = signals.iloc[i-1]['metadata']
    if isinstance(curr_meta, dict) and isinstance(prev_meta, dict):
        curr_close = curr_meta.get('close', 0)
        prev_close = prev_meta.get('close', 0)
        if prev_close > 0:
            ret = abs((curr_close - prev_close) / prev_close)
            returns.append(ret)

if returns:
    avg_move = np.mean(returns) * 100
    print(f"Average 5-min absolute move: {avg_move:.3f}%")
    print(f"Stop loss threshold: {stop_loss_pct * 100:.3f}%")
    print(f"Ratio (avg move / stop): {avg_move / (stop_loss_pct * 100):.2f}x")
    
    if avg_move > stop_loss_pct * 100 * 0.5:
        print("⚠️ Average 5-min move is significant relative to stop loss!")