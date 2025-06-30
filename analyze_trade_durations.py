#!/usr/bin/env python3
"""Analyze trade durations to understand overnight impact."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def analyze_trade_durations(signals_file: str, stop_pct: float = None):
    """Analyze how long trades are held."""
    
    # Load signals
    signals_df = pd.read_parquet(signals_file)
    if signals_df.empty:
        return []
    
    # Convert timestamp
    signals_df['datetime'] = pd.to_datetime(signals_df['ts'])
    
    # Track trades with detailed timing
    trades = []
    entry_price = None
    entry_signal = None
    entry_time = None
    entry_idx = None
    
    for idx, row in signals_df.iterrows():
        signal = row['val']
        price = row['px']
        current_time = row['datetime']
        
        # Check stops if in position
        if entry_price is not None and stop_pct:
            if entry_signal > 0:  # Long
                drawdown = (entry_price - price) / entry_price
                if drawdown > stop_pct:
                    exit_price = entry_price * (1 - stop_pct)
                    log_return = np.log(exit_price / entry_price) * entry_signal
                    duration = current_time - entry_time
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'duration_minutes': duration.total_seconds() / 60,
                        'duration_days': duration.days,
                        'log_return': log_return,
                        'stopped': True,
                        'overnight': (current_time.date() != entry_time.date())
                    })
                    entry_price = None
                    continue
            else:  # Short
                drawdown = (price - entry_price) / entry_price
                if drawdown > stop_pct:
                    exit_price = entry_price * (1 + stop_pct)
                    log_return = np.log(exit_price / entry_price) * entry_signal
                    duration = current_time - entry_time
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'duration_minutes': duration.total_seconds() / 60,
                        'duration_days': duration.days,
                        'log_return': log_return,
                        'stopped': True,
                        'overnight': (current_time.date() != entry_time.date())
                    })
                    entry_price = None
                    continue
        
        # Process signals
        if signal != 0 and entry_price is None:
            entry_price = price
            entry_signal = signal
            entry_time = current_time
            entry_idx = idx
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit
            log_return = np.log(price / entry_price) * entry_signal
            duration = current_time - entry_time
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'duration_minutes': duration.total_seconds() / 60,
                'duration_days': duration.days,
                'log_return': log_return,
                'stopped': False,
                'overnight': (current_time.date() != entry_time.date())
            })
            
            # Handle reversal
            if signal != 0:
                entry_price = price
                entry_signal = signal
                entry_time = current_time
            else:
                entry_price = None
    
    return trades


def main():
    workspace = "workspaces/signal_generation_5433aa9b"
    signal_files = list(Path(workspace).glob("traces/SPY_*/signals/keltner_bands/*.parquet"))[:10]
    
    print("=== TRADE DURATION ANALYSIS ===\n")
    
    # Analyze without stops
    all_trades_no_stops = []
    for signal_file in signal_files:
        trades = analyze_trade_durations(str(signal_file), stop_pct=None)
        all_trades_no_stops.extend(trades)
    
    # Analyze with stops
    all_trades_with_stops = []
    for signal_file in signal_files:
        trades = analyze_trade_durations(str(signal_file), stop_pct=0.003)
        all_trades_with_stops.extend(trades)
    
    # Convert to DataFrames
    df_no_stops = pd.DataFrame(all_trades_no_stops)
    df_with_stops = pd.DataFrame(all_trades_with_stops)
    
    print("WITHOUT STOPS:")
    if not df_no_stops.empty:
        print(f"Total trades: {len(df_no_stops)}")
        print(f"Average duration: {df_no_stops['duration_minutes'].mean():.1f} minutes")
        print(f"Median duration: {df_no_stops['duration_minutes'].median():.1f} minutes")
        print(f"Overnight trades: {df_no_stops['overnight'].sum()} ({df_no_stops['overnight'].mean()*100:.1f}%)")
        
        # Returns by overnight status
        overnight_trades = df_no_stops[df_no_stops['overnight']]
        intraday_trades = df_no_stops[~df_no_stops['overnight']]
        
        if len(overnight_trades) > 0:
            overnight_return = overnight_trades['log_return'].mean() * 10000 * 0.9998
            print(f"\nOvernight trades edge: {overnight_return:.2f} bps")
            print(f"Overnight avg duration: {overnight_trades['duration_minutes'].mean():.1f} min")
        
        if len(intraday_trades) > 0:
            intraday_return = intraday_trades['log_return'].mean() * 10000 * 0.9998
            print(f"Intraday trades edge: {intraday_return:.2f} bps")
            print(f"Intraday avg duration: {intraday_trades['duration_minutes'].mean():.1f} min")
    
    print("\n\nWITH 0.3% STOPS:")
    if not df_with_stops.empty:
        print(f"Total trades: {len(df_with_stops)}")
        print(f"Average duration: {df_with_stops['duration_minutes'].mean():.1f} minutes")
        print(f"Median duration: {df_with_stops['duration_minutes'].median():.1f} minutes")
        print(f"Overnight trades: {df_with_stops['overnight'].sum()} ({df_with_stops['overnight'].mean()*100:.1f}%)")
        print(f"Stopped trades: {df_with_stops['stopped'].sum()} ({df_with_stops['stopped'].mean()*100:.1f}%)")
        
        # Returns by type
        overnight_trades = df_with_stops[df_with_stops['overnight']]
        intraday_trades = df_with_stops[~df_with_stops['overnight']]
        stopped_trades = df_with_stops[df_with_stops['stopped']]
        normal_exits = df_with_stops[~df_with_stops['stopped']]
        
        if len(overnight_trades) > 0:
            overnight_return = overnight_trades['log_return'].mean() * 10000 * 0.9998
            print(f"\nOvernight trades edge: {overnight_return:.2f} bps")
            print(f"Overnight avg duration: {overnight_trades['duration_minutes'].mean():.1f} min")
        
        if len(intraday_trades) > 0:
            intraday_return = intraday_trades['log_return'].mean() * 10000 * 0.9998
            print(f"Intraday trades edge: {intraday_return:.2f} bps")
        
        if len(stopped_trades) > 0:
            stopped_return = stopped_trades['log_return'].mean() * 10000 * 0.9998
            print(f"\nStopped trades avg return: {stopped_return:.2f} bps")
            print(f"Stopped trades avg duration: {stopped_trades['duration_minutes'].mean():.1f} min")
        
        if len(normal_exits) > 0:
            normal_return = normal_exits['log_return'].mean() * 10000 * 0.9998
            print(f"Normal exit trades avg return: {normal_return:.2f} bps")
            print(f"Normal exit avg duration: {normal_exits['duration_minutes'].mean():.1f} min")
    
    # Duration distribution
    print("\n\nDURATION DISTRIBUTION (with stops):")
    if not df_with_stops.empty:
        bins = [0, 60, 240, 480, 960, 1440, 10000]  # minutes
        labels = ['<1hr', '1-4hr', '4-8hr', '8-16hr', '16-24hr', '>24hr']
        df_with_stops['duration_bin'] = pd.cut(df_with_stops['duration_minutes'], bins=bins, labels=labels)
        
        for label in labels:
            count = len(df_with_stops[df_with_stops['duration_bin'] == label])
            if count > 0:
                avg_return = df_with_stops[df_with_stops['duration_bin'] == label]['log_return'].mean() * 10000 * 0.9998
                print(f"{label:8s}: {count:4d} trades, {avg_return:6.2f} bps avg")


if __name__ == "__main__":
    main()