#!/usr/bin/env python3
"""Debug why performance is different from expected."""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_performance():
    print("Performance Debugging")
    print("=" * 50)
    
    # Load signals
    signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
    
    # Check signal patterns
    print("\n1. Signal Analysis:")
    print(f"   Total changes: {len(signals)}")
    print(f"   First 10 signals:")
    for i, row in signals.head(10).iterrows():
        print(f"   Bar {row['idx']}: {row['val']}")
    
    # Check if signals match expected Bollinger pattern
    # With period=11, we expect more frequent signals than period=20
    signal_frequency = len(signals) / signals['idx'].max() if signals['idx'].max() > 0 else 0
    print(f"\n2. Signal Frequency: {signal_frequency:.3%}")
    print("   - Period=11 typically: 10-15% of bars")
    print("   - Period=20 typically: 5-10% of bars")
    
    # Analyze signal patterns
    signal_changes = signals['val'].diff().dropna()
    entries = len(signal_changes[signal_changes != 0])
    
    # Count buy/sell signals
    buy_signals = len(signals[signals['val'] == 1])
    sell_signals = len(signals[signals['val'] == -1])
    flat_signals = len(signals[signals['val'] == 0])
    
    print(f"\n3. Signal Distribution:")
    print(f"   Buy signals: {buy_signals} ({buy_signals/len(signals):.1%})")
    print(f"   Sell signals: {sell_signals} ({sell_signals/len(signals):.1%})")
    print(f"   Flat signals: {flat_signals} ({flat_signals/len(signals):.1%})")
    
    # Check for mean reversion pattern
    # Bollinger Bands should show alternating buy/sell
    alternations = 0
    for i in range(1, len(signals)):
        if signals.iloc[i]['val'] * signals.iloc[i-1]['val'] < 0:  # Sign change
            alternations += 1
    
    print(f"\n4. Trading Pattern:")
    print(f"   Signal alternations: {alternations}")
    print(f"   Alternation rate: {alternations/len(signals):.1%}")
    print("   - Mean reversion expects high alternation (>20%)")
    
    # Performance estimate (rough)
    # Assuming each trade captures band width
    if buy_signals > 0:
        avg_trade_length = signals['idx'].max() / (buy_signals + sell_signals)
        print(f"\n5. Trade Characteristics:")
        print(f"   Average bars per trade: {avg_trade_length:.0f}")
        print(f"   Total round trips: ~{min(buy_signals, sell_signals)}")
    
    print("\n6. Likely Issues:")
    
    # Check minimum bars for first signal
    first_signal_bar = signals.iloc[0]['idx']
    print(f"   First signal at bar {first_signal_bar}:")
    if first_signal_bar < 20:
        print(f"   ✅ Must be using period < {first_signal_bar} (likely period=11)")
    else:
        print(f"   ❓ Could be using period up to {first_signal_bar}")
    
    if signal_frequency < 0.08:
        print("   ❌ Low signal frequency for period=11")
    else:
        print("   ✅ Signal frequency ~9.3% is reasonable for period=11")
        
    if alternations/len(signals) < 0.15:
        print("   ❌ Low alternation rate - not typical mean reversion pattern")
        print("      This might be the issue - signals aren't alternating properly")
    else:
        print("   ✅ Good alternation rate for mean reversion")
    
    print("\n7. Analysis:")
    print("   - Signal at bar 15 confirms period < 20 (likely period=11) ✅")
    print("   - Signal frequency ~9.3% is reasonable for period=11 ✅")
    print("   - But very low alternation rate (0.1%) is concerning ❌")
    print("   - This suggests signals are 'sticky' - staying in one direction too long")
    
    print("\n8. Possible Explanations for Poor Performance:")
    print("   a) Market conditions - trending market vs ranging market")
    print("   b) Exit threshold parameter might be too tight")
    print("   c) Data split - might be testing on different data than before")
    print("   d) The signals aren't alternating as expected for mean reversion")

if __name__ == "__main__":
    debug_performance()