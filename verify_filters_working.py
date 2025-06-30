#!/usr/bin/env python3
"""Verify that filters are actually being applied."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

def compare_signal_counts(workspace_path: str):
    """Compare signal counts to verify filters are working."""
    
    workspace = Path(workspace_path)
    signal_files = sorted(glob(str(workspace / "traces/SPY_*/signals/keltner_bands/*.parquet")))
    
    print("Verifying filter application by checking signal counts:\n")
    
    results = []
    for i, signal_file in enumerate(signal_files):
        signals_df = pd.read_parquet(signal_file)
        
        # Count non-zero signals
        non_zero_signals = len(signals_df[signals_df['val'] != 0])
        
        # Get total bars from first signal change to last
        if len(signals_df) > 0:
            first_ts = pd.to_datetime(signals_df['ts'].iloc[0])
            last_ts = pd.to_datetime(signals_df['ts'].iloc[-1])
            
            # For 5m data, estimate total bars
            time_span = (last_ts - first_ts).total_seconds() / 60  # minutes
            estimated_bars = time_span / 5  # 5-minute bars
        else:
            estimated_bars = 0
        
        results.append({
            'strategy_id': i,
            'signal_changes': len(signals_df),
            'non_zero_signals': non_zero_signals,
            'estimated_bars': estimated_bars
        })
    
    df = pd.DataFrame(results)
    
    # Group by expected filter types
    print("SIGNAL COUNTS BY STRATEGY GROUP:\n")
    
    # Base strategies (0-6) should have similar counts
    base_strategies = df[df['strategy_id'] <= 6]
    print(f"1. Base/Multiplier strategies (0-6):")
    print(f"   Avg signal changes: {base_strategies['signal_changes'].mean():.0f}")
    print(f"   Range: {base_strategies['signal_changes'].min()}-{base_strategies['signal_changes'].max()}")
    
    # VWAP filters (7-12) should have fewer signals
    vwap_strategies = df[(df['strategy_id'] >= 7) & (df['strategy_id'] <= 12)]
    print(f"\n2. VWAP filters (7-12):")
    print(f"   Avg signal changes: {vwap_strategies['signal_changes'].mean():.0f}")
    print(f"   Range: {vwap_strategies['signal_changes'].min()}-{vwap_strategies['signal_changes'].max()}")
    
    # RSI filters (13-21) should have much fewer signals
    rsi_strategies = df[(df['strategy_id'] >= 13) & (df['strategy_id'] <= 21)]
    print(f"\n3. RSI filters (13-21):")
    print(f"   Avg signal changes: {rsi_strategies['signal_changes'].mean():.0f}")
    print(f"   Range: {rsi_strategies['signal_changes'].min()}-{rsi_strategies['signal_changes'].max()}")
    
    # Volume filters (22-25)
    vol_strategies = df[(df['strategy_id'] >= 22) & (df['strategy_id'] <= 25)]
    print(f"\n4. Volume filters (22-25):")
    print(f"   Avg signal changes: {vol_strategies['signal_changes'].mean():.0f}")
    print(f"   Range: {vol_strategies['signal_changes'].min()}-{vol_strategies['signal_changes'].max()}")
    
    # Check for suspiciously similar counts
    print("\n\nFILTER EFFECTIVENESS CHECK:")
    
    # Calculate reduction from base
    base_avg = base_strategies['signal_changes'].mean()
    
    print(f"\nSignal reduction from base strategy:")
    print(f"VWAP filters: {(1 - vwap_strategies['signal_changes'].mean() / base_avg) * 100:.1f}% reduction")
    print(f"RSI filters: {(1 - rsi_strategies['signal_changes'].mean() / base_avg) * 100:.1f}% reduction")
    print(f"Volume filters: {(1 - vol_strategies['signal_changes'].mean() / base_avg) * 100:.1f}% reduction")
    
    # Look for identical signal counts (sign that filters aren't working)
    print("\n\nDUPLICATE SIGNAL COUNTS:")
    signal_count_freq = df['signal_changes'].value_counts()
    duplicates = signal_count_freq[signal_count_freq > 5]  # More than 5 with same count is suspicious
    
    if len(duplicates) > 0:
        print("WARNING: Many strategies have identical signal counts!")
        for count, freq in duplicates.items():
            print(f"  {freq} strategies have exactly {count} signals")
            matching_ids = df[df['signal_changes'] == count]['strategy_id'].tolist()
            print(f"    Strategy IDs: {matching_ids[:10]}...")  # Show first 10
    else:
        print("Good: Signal counts vary as expected")
    
    # Save detailed results
    df.to_csv("filter_verification.csv", index=False)
    print(f"\n\nDetailed results saved to filter_verification.csv")

if __name__ == "__main__":
    compare_signal_counts("workspaces/signal_generation_15c51c13")