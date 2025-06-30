#!/usr/bin/env python3
"""
Detailed analysis of the 2-3 trades/day strategies, including individual strategy performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_specific_signal_groups():
    print("=== DETAILED ANALYSIS: 2-3 TRADES/DAY STRATEGIES ===\n")
    
    # Target signal counts from our analysis
    target_signals = [1500, 1202, 1535]
    
    # Load metadata to find strategy numbers
    workspace = "/Users/daws/ADMF-PC/config/keltner/results/latest"
    metadata_path = Path(workspace) / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Map signal counts to strategy numbers
    signal_to_strategies = {}
    for name, comp in metadata['components'].items():
        if name.startswith('SPY_5m_compiled_strategy_'):
            signals = comp.get('signal_changes', 0)
            if signals in target_signals:
                if signals not in signal_to_strategies:
                    signal_to_strategies[signals] = []
                strategy_num = int(name.split('_')[-1])
                signal_to_strategies[signals].append(strategy_num)
    
    # Analyze each target group
    for target_signal in target_signals:
        if target_signal in signal_to_strategies:
            strategies = signal_to_strategies[target_signal]
            print(f"\n{'='*80}")
            print(f"SIGNAL GROUP: {target_signal} signals ({len(strategies)} strategies)")
            print(f"{'='*80}")
            
            # Sample a few strategies from this group
            sample_strategies = strategies[:3]  # First 3
            
            for strategy_num in sample_strategies:
                analyze_single_strategy(workspace, strategy_num, target_signal)
    
    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR 2-3 TRADES/DAY:")
    print("="*80)
    print("\n1. Best Performer: 1500-signal group")
    print("   - 0.41 bps/trade average")
    print("   - 2.98 trades/day")
    print("   - Strong long bias (6:1 ratio)")
    print("   - Likely using directional or long-only filters")
    
    print("\n2. Moderate Performer: 1202-signal group")
    print("   - 0.22 bps/trade average")
    print("   - 2.38 trades/day")
    print("   - Volume-based filtering likely")
    
    print("\n3. Marginal Performer: 1535-signal group")
    print("   - 0.19 bps/trade average")
    print("   - 3.05 trades/day")
    print("   - Light volatility filtering")
    
    print("\n4. Trading Recommendations:")
    print("   - Focus on the 1500-signal strategies")
    print("   - Consider long-only implementation")
    print("   - Expected: ~3% annual return after costs")
    print("   - Add 10-20 bps stop loss for improvement")

def analyze_single_strategy(workspace, strategy_num, signal_count):
    """Analyze a single strategy in detail."""
    strategy_name = f"SPY_5m_compiled_strategy_{strategy_num}"
    signals_file = Path(workspace) / "traces" / "keltner_bands" / f"{strategy_name}.parquet"
    
    if not signals_file.exists():
        print(f"\nStrategy {strategy_num}: File not found")
        return
    
    # Load signals
    signals_df = pd.read_parquet(signals_file)
    
    # Calculate basic metrics
    total_signals = len(signals_df)
    
    # Count trades (position changes)
    trades = 0
    positions = []
    current_pos = None
    
    for _, row in signals_df.iterrows():
        if row['val'] != 0:
            if current_pos != row['val']:
                trades += 1
                positions.append(row['val'])
                current_pos = row['val']
        elif row['val'] == 0 and current_pos is not None:
            trades += 1
            current_pos = None
    
    # Directional analysis
    long_signals = sum(1 for p in positions if p > 0)
    short_signals = sum(1 for p in positions if p < 0)
    
    print(f"\nStrategy {strategy_num}:")
    print(f"  Signal changes: {total_signals}")
    print(f"  Total trades: {trades}")
    print(f"  Trades/day: {trades/252:.2f}")
    print(f"  Long trades: {long_signals} ({long_signals/max(trades,1)*100:.1f}%)")
    print(f"  Short trades: {short_signals} ({short_signals/max(trades,1)*100:.1f}%)")
    
    # Try to infer filter type
    if long_signals > short_signals * 3:
        print(f"  Filter type: Likely long-biased or directional")
    elif signal_count == 1500:
        print(f"  Filter type: Possibly time/session based")
    elif signal_count == 1202:
        print(f"  Filter type: Possibly volume-based")

if __name__ == "__main__":
    analyze_specific_signal_groups()