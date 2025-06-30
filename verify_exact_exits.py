"""Verify that stops and targets are exiting at exact prices after the fix."""

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
            if exit_type == 'stop_loss':
                stop_losses.append({
                    'price': float(metadata.get('price', 0)),
                    'reason': nested.get('exit_reason', '')
                })
            elif exit_type == 'take_profit':
                take_profits.append({
                    'price': float(metadata.get('price', 0)),
                    'reason': nested.get('exit_reason', '')
                })

print(f"\nStop loss exits: {len(stop_losses)}")
print(f"Take profit exits: {len(take_profits)}")

# Analyze stop losses
if stop_losses:
    print("\n=== STOP LOSS ANALYSIS ===")
    
    # Find pairs of entry/exit to calculate exact returns
    exact_stop_count = 0
    total_analyzed = 0
    
    for i in range(1, len(fills_df)):
        row = fills_df.iloc[i]
        metadata = row['metadata']
        if isinstance(metadata, dict):
            nested = metadata.get('metadata', {})
            if nested.get('exit_type') == 'stop_loss':
                exit_price = float(metadata.get('price', 0))
                
                # Find the previous entry
                for j in range(i-1, -1, -1):
                    prev_row = fills_df.iloc[j]
                    prev_meta = prev_row['metadata']
                    if isinstance(prev_meta, dict):
                        prev_nested = prev_meta.get('metadata', {})
                        # Check if this is an entry (no exit_type)
                        if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                            entry_price = float(prev_meta.get('price', 0))
                            if entry_price > 0:
                                return_pct = (exit_price - entry_price) / entry_price * 100
                                total_analyzed += 1
                                
                                # Check if it's exactly -0.075%
                                if abs(return_pct - (-0.075)) < 0.001:
                                    exact_stop_count += 1
                                
                                if total_analyzed <= 5:
                                    print(f"\nExample {total_analyzed}:")
                                    print(f"  Entry: ${entry_price:.4f}")
                                    print(f"  Exit:  ${exit_price:.4f}")
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
    
    for i in range(1, len(fills_df)):
        row = fills_df.iloc[i]
        metadata = row['metadata']
        if isinstance(metadata, dict):
            nested = metadata.get('metadata', {})
            if nested.get('exit_type') == 'take_profit':
                exit_price = float(metadata.get('price', 0))
                
                # Find the previous entry
                for j in range(i-1, -1, -1):
                    prev_row = fills_df.iloc[j]
                    prev_meta = prev_row['metadata']
                    if isinstance(prev_meta, dict):
                        prev_nested = prev_meta.get('metadata', {})
                        # Check if this is an entry (no exit_type)
                        if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                            entry_price = float(prev_meta.get('price', 0))
                            if entry_price > 0:
                                return_pct = (exit_price - entry_price) / entry_price * 100
                                total_analyzed += 1
                                
                                # Check if it's exactly 0.15%
                                if abs(return_pct - 0.15) < 0.001:
                                    exact_take_count += 1
                                
                                if total_analyzed <= 5:
                                    print(f"\nExample {total_analyzed}:")
                                    print(f"  Entry: ${entry_price:.4f}")
                                    print(f"  Exit:  ${exit_price:.4f}")
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
print("\nWith the fix, stops and takes should now exit at exact prices!")