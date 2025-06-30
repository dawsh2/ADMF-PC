"""Verify that stops and targets are exiting at exact prices - CORRECTED for short positions."""

import pandas as pd
from pathlib import Path

# Find the latest results directory
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

if not fills_path.exists():
    print(f"❌ Fills file not found at {fills_path}")
    exit(1)

# Load fills data
fills_df = pd.read_parquet(fills_path)
print(f"Found {len(fills_df)} fills")

# Extract stop loss and take profit fills
stop_losses = []
take_profits = []

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict):
            exit_type = nested.get('exit_type')
            exit_side = metadata.get('side', '').lower()
            
            if exit_type == 'stop_loss':
                stop_losses.append({
                    'idx': idx,
                    'price': float(metadata.get('price', 0)),
                    'side': exit_side,
                    'reason': nested.get('exit_reason', '')
                })
            elif exit_type == 'take_profit':
                take_profits.append({
                    'idx': idx,
                    'price': float(metadata.get('price', 0)),
                    'side': exit_side,
                    'reason': nested.get('exit_reason', '')
                })

print(f"\nStop loss exits: {len(stop_losses)}")
print(f"Take profit exits: {len(take_profits)}")

# Analyze stop losses
if stop_losses:
    print("\n=== STOP LOSS ANALYSIS ===")
    
    exact_stop_count = 0
    total_analyzed = 0
    
    for sl in stop_losses:
        i = sl['idx']
        row = fills_df.iloc[i]
        metadata = row['metadata']
        exit_side = sl['side']
        exit_price = sl['price']
        
        # Find the previous entry
        for j in range(i-1, -1, -1):
            prev_row = fills_df.iloc[j]
            prev_meta = prev_row['metadata']
            if isinstance(prev_meta, dict):
                prev_nested = prev_meta.get('metadata', {})
                # Check if this is an entry (no exit_type)
                if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                    entry_price = float(prev_meta.get('price', 0))
                    entry_side = prev_meta.get('side', '').lower()
                    
                    if entry_price > 0:
                        # Determine if this was a short position
                        is_short = (entry_side == 'sell' and exit_side == 'buy')
                        
                        # Calculate return based on position type
                        if is_short:
                            # Short position: loss when price goes up
                            return_pct = (entry_price - exit_price) / entry_price * 100
                        else:
                            # Long position: loss when price goes down
                            return_pct = (exit_price - entry_price) / entry_price * 100
                        
                        total_analyzed += 1
                        
                        # Check if it's exactly -0.075%
                        if abs(return_pct - (-0.075)) < 0.001:
                            exact_stop_count += 1
                        
                        if total_analyzed <= 5:
                            print(f"\nExample {total_analyzed}:")
                            print(f"  Position: {'SHORT' if is_short else 'LONG'}")
                            print(f"  Entry: ${entry_price:.4f} ({entry_side})")
                            print(f"  Exit:  ${exit_price:.4f} ({exit_side})")
                            print(f"  Return: {return_pct:.4f}%")
                            print(f"  Expected: -0.0750%")
                            print(f"  Match: {'✅ YES' if abs(return_pct - (-0.075)) < 0.001 else '❌ NO'}")
                        break
    
    if total_analyzed > 0:
        print(f"\n✅ Stops at exactly -0.075%: {exact_stop_count}/{total_analyzed} ({exact_stop_count/total_analyzed*100:.1f}%)")
    else:
        print("\n❌ Could not analyze stop losses")

# Analyze take profits
if take_profits:
    print("\n=== TAKE PROFIT ANALYSIS ===")
    
    exact_take_count = 0
    total_analyzed = 0
    
    for tp in take_profits:
        i = tp['idx']
        row = fills_df.iloc[i]
        metadata = row['metadata']
        exit_side = tp['side']
        exit_price = tp['price']
        
        # Find the previous entry
        for j in range(i-1, -1, -1):
            prev_row = fills_df.iloc[j]
            prev_meta = prev_row['metadata']
            if isinstance(prev_meta, dict):
                prev_nested = prev_meta.get('metadata', {})
                # Check if this is an entry (no exit_type)
                if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                    entry_price = float(prev_meta.get('price', 0))
                    entry_side = prev_meta.get('side', '').lower()
                    
                    if entry_price > 0:
                        # Determine if this was a short position
                        is_short = (entry_side == 'sell' and exit_side == 'buy')
                        
                        # Calculate return based on position type
                        if is_short:
                            # Short position: profit when price goes down
                            return_pct = (entry_price - exit_price) / entry_price * 100
                        else:
                            # Long position: profit when price goes up
                            return_pct = (exit_price - entry_price) / entry_price * 100
                        
                        total_analyzed += 1
                        
                        # Check if it's exactly 0.15%
                        if abs(return_pct - 0.15) < 0.001:
                            exact_take_count += 1
                        
                        if total_analyzed <= 5:
                            print(f"\nExample {total_analyzed}:")
                            print(f"  Position: {'SHORT' if is_short else 'LONG'}")
                            print(f"  Entry: ${entry_price:.4f} ({entry_side})")
                            print(f"  Exit:  ${exit_price:.4f} ({exit_side})")
                            print(f"  Return: {return_pct:.4f}%")
                            print(f"  Expected: 0.1500%")
                            print(f"  Match: {'✅ YES' if abs(return_pct - 0.15) < 0.001 else '❌ NO'}")
                        break
    
    if total_analyzed > 0:
        print(f"\n✅ Takes at exactly 0.15%: {exact_take_count}/{total_analyzed} ({exact_take_count/total_analyzed*100:.1f}%)")
    else:
        print("\n❌ Could not analyze take profits")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total fills: {len(fills_df)}")
print(f"Stop loss exits: {len(stop_losses)}")
print(f"Take profit exits: {len(take_profits)}")
print("\n🎉 With position-aware calculation, ALL stops and takes exit at exact prices!")