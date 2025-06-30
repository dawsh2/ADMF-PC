#!/usr/bin/env python3
"""
Check how entries are handled in the notebook
"""
import pandas as pd
import numpy as np

# Load your data (adjust this to match your notebook's data loading)
# Assuming df is already loaded with columns: timestamp, open, high, low, close, signal
# df = pd.read_csv(...) or however you load it

def analyze_entry_logic(df):
    """Analyze how entries would be handled with different methods"""
    
    print("=== Notebook Entry Logic Analysis ===\n")
    
    # Find first few signal entries
    entry_examples = []
    for i in range(1, len(df)-1):
        if df.iloc[i-1]['signal'] == 0 and df.iloc[i]['signal'] != 0:
            entry_examples.append(i)
            if len(entry_examples) >= 3:  # Get 3 examples
                break
    
    if not entry_examples:
        print("No entry signals found in the data!")
        return
    
    # Analyze each example
    for idx, i in enumerate(entry_examples):
        print(f"\n--- Entry Example {idx+1} ---")
        print(f"Signal changes from 0 to {df.iloc[i]['signal']} at index {i}")
        
        # Show the bars
        prev_bar = df.iloc[i-1]
        signal_bar = df.iloc[i]
        next_bar = df.iloc[i+1]
        
        print(f"\nPrevious bar (i-1):")
        print(f"  Time: {prev_bar['timestamp']}")
        print(f"  OHLC: {prev_bar['open']:.2f} / {prev_bar['high']:.2f} / {prev_bar['low']:.2f} / {prev_bar['close']:.2f}")
        print(f"  Signal: {prev_bar['signal']}")
        
        print(f"\nSignal bar (i):")
        print(f"  Time: {signal_bar['timestamp']}")
        print(f"  OHLC: {signal_bar['open']:.2f} / {signal_bar['high']:.2f} / {signal_bar['low']:.2f} / {signal_bar['close']:.2f}")
        print(f"  Signal: {signal_bar['signal']}")
        
        print(f"\nNext bar (i+1):")
        print(f"  Time: {next_bar['timestamp']}")
        print(f"  OHLC: {next_bar['open']:.2f} / {next_bar['high']:.2f} / {next_bar['low']:.2f} / {next_bar['close']:.2f}")
        print(f"  Signal: {next_bar['signal']}")
        
        # Calculate entry prices
        close_entry = signal_bar['close']
        open_entry = next_bar['open']
        
        print(f"\n**Entry Price Options:**")
        print(f"  Option 1 - Close of signal bar: ${close_entry:.2f}")
        print(f"  Option 2 - Open of next bar:    ${open_entry:.2f}")
        print(f"  Difference: ${abs(open_entry - close_entry):.2f}")
        
        # Check stop loss impact
        stop_loss_pct = 0.00075  # 0.075%
        direction = "LONG" if signal_bar['signal'] > 0 else "SHORT"
        
        print(f"\n**Stop Loss Analysis ({direction} position):**")
        
        if direction == "LONG":
            stop_from_close = close_entry * (1 - stop_loss_pct)
            stop_from_open = open_entry * (1 - stop_loss_pct)
            
            print(f"  Stop from close entry: ${stop_from_close:.2f}")
            print(f"  Stop from open entry:  ${stop_from_open:.2f}")
            
            # Check if next bar would hit stop
            if next_bar['low'] <= stop_from_close:
                print(f"  ⚠️  Next bar low (${next_bar['low']:.2f}) would hit stop from close entry!")
            else:
                print(f"  ✓  Next bar low (${next_bar['low']:.2f}) above stop from close entry")
                
            if next_bar['low'] <= stop_from_open:
                print(f"  ⚠️  Next bar low (${next_bar['low']:.2f}) would hit stop from open entry!")
            else:
                print(f"  ✓  Next bar low (${next_bar['low']:.2f}) above stop from open entry")
        else:  # SHORT
            stop_from_close = close_entry * (1 + stop_loss_pct)
            stop_from_open = open_entry * (1 + stop_loss_pct)
            
            print(f"  Stop from close entry: ${stop_from_close:.2f}")
            print(f"  Stop from open entry:  ${stop_from_open:.2f}")
            
            # Check if next bar would hit stop
            if next_bar['high'] >= stop_from_close:
                print(f"  ⚠️  Next bar high (${next_bar['high']:.2f}) would hit stop from close entry!")
            else:
                print(f"  ✓  Next bar high (${next_bar['high']:.2f}) below stop from close entry")
                
            if next_bar['high'] >= stop_from_open:
                print(f"  ⚠️  Next bar high (${next_bar['high']:.2f}) would hit stop from open entry!")
            else:
                print(f"  ✓  Next bar high (${next_bar['high']:.2f}) below stop from open entry")

    # Summary statistics
    print("\n\n=== Summary Statistics ===")
    
    close_stops = 0
    open_stops = 0
    total_entries = 0
    
    for i in range(1, len(df)-1):
        if df.iloc[i-1]['signal'] == 0 and df.iloc[i]['signal'] != 0:
            total_entries += 1
            signal_bar = df.iloc[i]
            next_bar = df.iloc[i+1]
            
            close_entry = signal_bar['close']
            open_entry = next_bar['open']
            
            if signal_bar['signal'] > 0:  # LONG
                stop_from_close = close_entry * (1 - stop_loss_pct)
                stop_from_open = open_entry * (1 - stop_loss_pct)
                
                if next_bar['low'] <= stop_from_close:
                    close_stops += 1
                if next_bar['low'] <= stop_from_open:
                    open_stops += 1
            else:  # SHORT
                stop_from_close = close_entry * (1 + stop_loss_pct)
                stop_from_open = open_entry * (1 + stop_loss_pct)
                
                if next_bar['high'] >= stop_from_close:
                    close_stops += 1
                if next_bar['high'] >= stop_from_open:
                    open_stops += 1
    
    print(f"Total entry signals analyzed: {total_entries}")
    print(f"\nImmediate stops (next bar):")
    print(f"  Entering at close: {close_stops} ({close_stops/total_entries*100:.1f}%)")
    print(f"  Entering at open:  {open_stops} ({open_stops/total_entries*100:.1f}%)")
    print(f"  Difference: {close_stops - open_stops} fewer stops with open entry")
    
    print("\n=== Recommendation ===")
    print("If your notebook shows significantly fewer stop losses than")
    print("the backtest, it's likely using the OPEN of the next bar")
    print("for entry, not the CLOSE of the signal bar.")

# If running standalone, you'll need to load your data first
if __name__ == "__main__":
    print("Please load your data into a DataFrame 'df' first, then call:")
    print("  analyze_entry_logic(df)")
    print("\nRequired columns: timestamp, open, high, low, close, signal")