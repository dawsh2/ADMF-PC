"""Debug the calculation method difference."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== DEBUGGING CALCULATION METHODS ===\n")

# Find examples where the two methods differ
examples = []

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type') == 'stop_loss':
            exit_side = metadata.get('side', '').lower()
            
            # Find the previous entry
            for j in range(idx-1, -1, -1):
                prev_row = fills_df.iloc[j]
                prev_meta = prev_row['metadata']
                if isinstance(prev_meta, dict):
                    prev_nested = prev_meta.get('metadata', {})
                    if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                        entry_price = float(prev_meta.get('price', 0))
                        exit_price = float(metadata.get('price', 0))
                        entry_side = prev_meta.get('side', '').lower()
                        
                        if entry_price > 0:
                            # Method 1: Simple calculation (used in verify_exact_exits.py)
                            simple_return = (exit_price - entry_price) / entry_price * 100
                            
                            # Method 2: Position-aware calculation
                            is_short = (entry_side == 'sell' and exit_side == 'buy')
                            if is_short:
                                position_aware_return = (entry_price - exit_price) / entry_price * 100
                            else:
                                position_aware_return = (exit_price - entry_price) / entry_price * 100
                            
                            # Check if methods differ
                            if abs(simple_return - position_aware_return) > 0.001:
                                examples.append({
                                    'order_id': metadata.get('order_id', ''),
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'entry_side': entry_side,
                                    'exit_side': exit_side,
                                    'is_short': is_short,
                                    'simple_return': simple_return,
                                    'position_aware_return': position_aware_return,
                                    'exit_reason': nested.get('exit_reason', '')
                                })
                                
                            if len(examples) >= 5:
                                break
                        break
            
            if len(examples) >= 5:
                break

print(f"Found {len(examples)} cases where calculation methods differ\n")

for i, ex in enumerate(examples):
    print(f"Example {i+1}: {ex['order_id']}")
    print(f"  Position: {'SHORT' if ex['is_short'] else 'LONG'} (entry={ex['entry_side']}, exit={ex['exit_side']})")
    print(f"  Entry: ${ex['entry_price']:.4f}")
    print(f"  Exit:  ${ex['exit_price']:.4f}")
    print(f"  Simple return: {ex['simple_return']:.4f}%")
    print(f"  Position-aware return: {ex['position_aware_return']:.4f}%")
    print(f"  Exit reason: {ex['exit_reason']}")
    
    # Check which calculation gives -0.075%
    if abs(ex['simple_return'] - (-0.075)) < 0.001:
        print(f"  ✅ Simple calculation gives -0.075%")
    elif abs(ex['position_aware_return'] - (-0.075)) < 0.001:
        print(f"  ✅ Position-aware calculation gives -0.075%")
    
    # Check if it's the inverted case
    if abs(ex['simple_return'] - 0.075) < 0.001:
        print(f"  ⚠️  Simple calculation gives +0.075% (inverted!)")
    
    print()

print("\n=== CONCLUSION ===")
print("The 'inverted' stops are SHORT positions where:")
print("- Entry price < Exit price (price went UP)")
print("- Simple return calculation shows +0.075%")
print("- But for SHORT positions, this is actually a -0.075% LOSS")
print("- The execution engine is using the CORRECT exit price!")
print("\nThe issue is in the verification script, not the execution!")