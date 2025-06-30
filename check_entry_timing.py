#!/usr/bin/env python3
"""
Check if we're entering at close vs open price
"""
import pandas as pd

# Read positions and signals
positions_open = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/portfolio/positions_open/positions_open.parquet')
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print("=== Entry Price Analysis ===\n")

# Check first 10 positions
for i in range(min(10, len(positions_open))):
    pos = positions_open.iloc[i]
    pos_meta = pos['metadata']
    
    if isinstance(pos_meta, dict):
        entry_price = pos_meta.get('entry_price', 0)
        entry_bar = pos['idx']
        
        # Find the signal bar
        signal_bar = signals[signals['idx'] == entry_bar]
        if len(signal_bar) > 0:
            sig_meta = signal_bar.iloc[0]['metadata']
            if isinstance(sig_meta, dict):
                print(f"Position {i+1}:")
                print(f"  Entry bar: {entry_bar}")
                print(f"  Entry price: ${entry_price:.2f}")
                print(f"  Bar OHLC: O=${sig_meta.get('open', 0):.2f}, H=${sig_meta.get('high', 0):.2f}, "
                      f"L=${sig_meta.get('low', 0):.2f}, C=${sig_meta.get('close', 0):.2f}")
                
                # Check which price we entered at
                open_price = sig_meta.get('open', 0)
                close_price = sig_meta.get('close', 0)
                
                if abs(entry_price - close_price) < 0.01:
                    print(f"  → Entered at CLOSE price")
                elif abs(entry_price - open_price) < 0.01:
                    print(f"  → Entered at OPEN price")
                else:
                    print(f"  → Entered at OTHER price (slippage?)")
                
                # Check the previous bar to see if signal came from there
                prev_bar = signals[signals['idx'] == entry_bar - 1]
                if len(prev_bar) > 0:
                    prev_sig = prev_bar.iloc[0]
                    print(f"  Previous bar signal: {prev_sig['val']}")
                    
                    # If we got signal on previous bar and entered at current bar's open,
                    # that would be more realistic
                    if prev_sig['val'] != 0:
                        print(f"  → Signal from previous bar, entry at current open would be realistic")
                
                print()

print("\n=== Key Question ===")
print("If we're entering at the CLOSE of the signal bar, but the notebook")
print("enters at the OPEN of the NEXT bar, that could explain the difference.")
print("The open of the next bar gives more room before hitting stops.")