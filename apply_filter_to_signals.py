#!/usr/bin/env python3
"""
Apply volatility filter to already generated signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def apply_volatility_filter():
    print("=== APPLYING VOLATILITY FILTER TO SIGNALS ===\n")
    
    # Load the generated signals
    signal_path = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/results/latest/traces/keltner_bands/SPY_5m_compiled_strategy_0.parquet")
    
    if not signal_path.exists():
        print(f"Signal file not found: {signal_path}")
        return
        
    signals_df = pd.read_parquet(signal_path)
    print(f"Loaded {len(signals_df)} signal changes")
    print(f"Signal values: {signals_df['val'].unique()}")
    
    # We need the actual market data to calculate ATR ratios
    data_paths = [
        "/Users/daws/ADMF-PC/data/SPY_5m_test.csv",
        "/Users/daws/ADMF-PC/data/SPY_test_5m.csv", 
        "/Users/daws/ADMF-PC/data/SPY_5m.csv"
    ]
    
    market_data = None
    for path in data_paths:
        if Path(path).exists():
            market_data = pd.read_csv(path)
            print(f"\nLoaded market data from: {path}")
            print(f"Data shape: {market_data.shape}")
            break
    
    if market_data is None:
        print("Could not find market data to calculate ATR")
        return
    
    # Calculate ATR values
    print("\nCalculating ATR values...")
    
    # True Range calculation
    market_data['prev_close'] = market_data['close'].shift(1)
    market_data['hl'] = market_data['high'] - market_data['low']
    market_data['hc'] = abs(market_data['high'] - market_data['prev_close'])
    market_data['lc'] = abs(market_data['low'] - market_data['prev_close'])
    market_data['tr'] = market_data[['hl', 'hc', 'lc']].max(axis=1)
    
    # Calculate ATRs
    market_data['atr_14'] = market_data['tr'].rolling(14).mean()
    market_data['atr_50'] = market_data['tr'].rolling(50).mean()
    market_data['vol_ratio'] = market_data['atr_14'] / market_data['atr_50']
    
    # Apply different thresholds
    print("\nApplying volatility filters:")
    print("-" * 60)
    
    thresholds = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for threshold in thresholds:
        # Get indices where volatility ratio > threshold
        high_vol_mask = market_data['vol_ratio'] > threshold
        high_vol_indices = market_data.index[high_vol_mask].tolist()
        
        # Filter signals to only those during high volatility
        # Signals are stored sparsely, so we need to reconstruct full signal array
        full_signals = np.zeros(len(market_data))
        
        # Fill in signal values
        for i in range(len(signals_df)):
            start_idx = signals_df.iloc[i]['idx']
            signal_value = signals_df.iloc[i]['val']
            
            # Find end index (next change or end of data)
            if i < len(signals_df) - 1:
                end_idx = signals_df.iloc[i + 1]['idx']
            else:
                end_idx = len(market_data)
            
            full_signals[start_idx:end_idx] = signal_value
        
        # Apply volatility filter
        filtered_signals = full_signals.copy()
        filtered_signals[~high_vol_mask] = 0  # Set to 0 when volatility is low
        
        # Count non-zero signals
        original_count = np.sum(full_signals != 0)
        filtered_count = np.sum(filtered_signals != 0)
        reduction = (1 - filtered_count / original_count) * 100 if original_count > 0 else 0
        
        print(f"\nThreshold {threshold}:")
        print(f"  Original signals: {original_count}")
        print(f"  Filtered signals: {filtered_count}")
        print(f"  Reduction: {reduction:.1f}%")
        
        # Count signal changes (for sparse representation)
        signal_changes = 0
        prev_signal = 0
        for i in range(len(filtered_signals)):
            if filtered_signals[i] != prev_signal:
                signal_changes += 1
                prev_signal = filtered_signals[i]
        
        print(f"  Signal changes: {signal_changes}")
        
        # Check if this matches expected 2826 pattern
        if threshold == 1.1:
            print(f"\n  Expected for 1.1 threshold: ~590 signal changes")
            print(f"  Actual: {signal_changes} signal changes")
    
    print("\n" + "="*60)
    print("INSIGHTS:")
    print("="*60)
    print("1. The volatility filter CAN be applied post-hoc")
    print("2. Original analysis likely used this approach")
    print("3. Filter reduces signals significantly")
    print("4. Test data may need different threshold than training")

if __name__ == "__main__":
    apply_volatility_filter()