#!/usr/bin/env python3
"""Analyze returns with various maximum holding periods + stops + EOD."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

def simulate_with_max_hold(signals_file: str, max_minutes: int, stop_pct: float = 0.003):
    """Simulate trades with maximum hold time in minutes, stops, and EOD exits."""
    
    signals_df = pd.read_parquet(signals_file)
    if signals_df.empty:
        return []
    
    # Convert timestamp
    signals_df['datetime'] = pd.to_datetime(signals_df['ts'])
    signals_df['date'] = signals_df['datetime'].dt.date
    signals_df['time'] = signals_df['datetime'].dt.time
    
    # Define market hours
    market_open = time(9, 30)
    market_close = time(15, 45)  # Stop new trades 15 min before close
    eod_close = time(15, 59)     # Force close all positions
    
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
        
        # Check exit conditions if in position
        if entry_price is not None:
            exit_reason = None
            exit_price = price
            
            # Priority 1: EOD force close
            if current_date != entry_date or current_tod >= eod_close:
                exit_reason = 'eod'
            
            # Priority 2: Max hold time
            elif max_minutes is not None:
                minutes_held = (current_time - entry_time).total_seconds() / 60
                if minutes_held >= max_minutes:
                    exit_reason = 'max_hold'
            
            # Priority 3: Stop loss
            if exit_reason is None and stop_pct is not None:
                if entry_signal > 0:  # Long
                    drawdown = (entry_price - price) / entry_price
                    if drawdown > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 - stop_pct)
                else:  # Short
                    drawdown = (price - entry_price) / entry_price
                    if drawdown > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 + stop_pct)
            
            # Priority 4: Normal signal exit
            if exit_reason is None and (signal == 0 or signal == -entry_signal):
                exit_reason = 'signal'
            
            # Execute exit if triggered
            if exit_reason:
                log_return = np.log(exit_price / entry_price) * entry_signal
                duration = (current_time - entry_time).total_seconds() / 60
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'duration_minutes': duration,
                    'log_return': log_return,
                    'exit_reason': exit_reason
                })
                
                # Clear position
                entry_price = None
                entry_signal = None
                
                # Enter new position if signal reversal (not near close)
                if signal != 0 and exit_reason == 'signal' and current_tod < market_close:
                    entry_price = price
                    entry_signal = signal
                    entry_time = current_time
                    entry_date = current_date
        
        # Enter new position if signaled (not near close)
        elif signal != 0 and entry_price is None and current_tod < market_close:
            entry_price = price
            entry_signal = signal
            entry_time = current_time
            entry_date = current_date
    
    return trades


def main():
    workspace = "workspaces/signal_generation_5433aa9b"
    signal_files = list(Path(workspace).glob("traces/SPY_*/signals/keltner_bands/*.parquet"))[:10]
    
    print("=== OPTIMAL MAXIMUM HOLD TIME ANALYSIS ===")
    print("All scenarios include: 0.3% stops + EOD exits + No trades after 3:45pm\n")
    
    # Test different max hold periods
    max_hold_minutes = [15, 30, 45, 60, 90, 120, 180, 240, None]  # None = no limit
    
    results = []
    
    for max_mins in max_hold_minutes:
        all_trades = []
        
        for signal_file in signal_files:
            trades = simulate_with_max_hold(str(signal_file), max_mins, stop_pct=0.003)
            all_trades.extend(trades)
        
        if all_trades:
            # Calculate metrics
            returns = [t['log_return'] * 0.9998 for t in all_trades]  # Apply costs
            edge_bps = np.mean(returns) * 10000
            total_return_bps = np.sum(returns) * 10000
            
            # Count exit reasons
            exit_counts = {}
            for t in all_trades:
                reason = t['exit_reason']
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
            
            # Duration stats
            durations = [t['duration_minutes'] for t in all_trades]
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            
            results.append({
                'max_hold': max_mins,
                'edge_bps': edge_bps,
                'total_trades': len(all_trades),
                'total_return_bps': total_return_bps,
                'avg_duration': avg_duration,
                'max_duration': max_duration,
                'pct_max_hold': exit_counts.get('max_hold', 0) / len(all_trades) * 100,
                'pct_stopped': exit_counts.get('stop', 0) / len(all_trades) * 100,
                'pct_eod': exit_counts.get('eod', 0) / len(all_trades) * 100,
                'pct_signal': exit_counts.get('signal', 0) / len(all_trades) * 100
            })
    
    # Display results
    print("Max Hold | Edge  | Trades | Avg Dur | Max Dur | Signal% | MaxHold% | Stop% | EOD%")
    print("---------|-------|--------|---------|---------|---------|----------|-------|-----")
    
    for r in results:
        max_hold_str = f"{r['max_hold']} min" if r['max_hold'] else "None"
        print(f"{max_hold_str:8s} | {r['edge_bps']:5.2f} | {r['total_trades']:6d} | "
              f"{r['avg_duration']:7.0f} | {r['max_duration']:7.0f} | "
              f"{r['pct_signal']:7.1f} | {r['pct_max_hold']:8.1f} | "
              f"{r['pct_stopped']:5.1f} | {r['pct_eod']:4.1f}")
    
    # Find optimal
    best = max(results[:-1], key=lambda x: x['edge_bps'])  # Exclude "None" from best
    baseline = results[-1]['edge_bps']  # No limit baseline
    
    print(f"\n\nOPTIMAL MAX HOLD: {best['max_hold']} minutes")
    print(f"Edge: {best['edge_bps']:.2f} bps")
    print(f"Improvement over no limit: {best['edge_bps'] - baseline:.2f} bps")
    print(f"Trades cut by max hold: {best['pct_max_hold']:.1f}%")
    
    # Show return curve
    print("\n\nRETURN CURVE BY MAX HOLD TIME:")
    print("Minutes | Edge (bps) | Cumulative Improvement")
    print("--------|------------|----------------------")
    
    for r in results[:-1]:  # Exclude "None"
        improvement = r['edge_bps'] - baseline
        bar_length = int(max(0, improvement * 5))  # Scale for display
        bar = '█' * bar_length if improvement > 0 else '▒' * abs(bar_length)
        print(f"{r['max_hold']:7d} | {r['edge_bps']:10.2f} | {improvement:+6.2f} {bar}")
    
    # Detailed breakdown of best scenario
    print(f"\n\nDETAILED BREAKDOWN ({best['max_hold']} minute max hold):")
    print(f"Total trades: {best['total_trades']}")
    print(f"Average duration: {best['avg_duration']:.0f} minutes")
    print(f"Maximum duration: {best['max_duration']:.0f} minutes (capped)")
    print("\nExit reasons:")
    print(f"  Normal signal exits: {best['pct_signal']:.1f}%")
    print(f"  Hit max hold time: {best['pct_max_hold']:.1f}%")
    print(f"  Stopped out (0.3%): {best['pct_stopped']:.1f}%")
    print(f"  End of day exits: {best['pct_eod']:.1f}%")
    print("\nNo overnight risk!")


if __name__ == "__main__":
    main()