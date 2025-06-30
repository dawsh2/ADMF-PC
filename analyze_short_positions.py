"""Analyze if the inverted stops are all for short positions."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== ANALYZING SHORT POSITIONS ===\n")

# Track all stop losses
long_stops = []
short_stops = []

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
                            # For short positions: entry=SELL, exit=BUY
                            is_short = (entry_side == 'sell' and exit_side == 'buy')
                            
                            # Calculate return based on position type
                            if is_short:
                                # Short position: profit when price goes down
                                return_pct = (entry_price - exit_price) / entry_price * 100
                            else:
                                # Long position: profit when price goes up
                                return_pct = (exit_price - entry_price) / entry_price * 100
                            
                            is_exact = abs(return_pct - (-0.075)) < 0.001
                            
                            stop_data = {
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'return_pct': return_pct,
                                'is_exact': is_exact,
                                'entry_side': entry_side,
                                'exit_side': exit_side,
                                'order_id': metadata.get('order_id', '')
                            }
                            
                            if is_short:
                                short_stops.append(stop_data)
                            else:
                                long_stops.append(stop_data)
                            break

print(f"Total stop losses: {len(long_stops) + len(short_stops)}")
print(f"Long position stops: {len(long_stops)}")
print(f"Short position stops: {len(short_stops)}")

# Analyze accuracy by position type
long_exact = sum(1 for s in long_stops if s['is_exact'])
short_exact = sum(1 for s in short_stops if s['is_exact'])

print(f"\nLong stops at exact -0.075%: {long_exact}/{len(long_stops)} ({long_exact/len(long_stops)*100:.1f}% if long_stops else 0)")
print(f"Short stops at exact -0.075%: {short_exact}/{len(short_stops)} ({short_exact/len(short_stops)*100:.1f}% if short_stops else 0)")

# Check for inverted returns
print("\n=== INVERTED RETURNS ANALYSIS ===")

# For long positions, inverted = positive return
long_inverted = [s for s in long_stops if abs(s['return_pct'] - 0.075) < 0.01]
print(f"\nLong positions with inverted returns (+0.075%): {len(long_inverted)}")
if long_inverted:
    for i, inv in enumerate(long_inverted[:3]):
        print(f"  Entry: ${inv['entry_price']:.4f} ({inv['entry_side']}), Exit: ${inv['exit_price']:.4f} ({inv['exit_side']}), Return: {inv['return_pct']:.4f}%")

# For short positions, check if the calculation is wrong
short_wrong = [s for s in short_stops if not s['is_exact']]
print(f"\nShort positions with wrong returns: {len(short_wrong)}")
if short_wrong:
    for i, wrong in enumerate(short_wrong[:3]):
        print(f"  Entry: ${wrong['entry_price']:.4f} ({wrong['entry_side']}), Exit: ${wrong['exit_price']:.4f} ({wrong['exit_side']}), Return: {wrong['return_pct']:.4f}%")
        # Check what the return would be with wrong calculation
        wrong_calc = (wrong['exit_price'] - wrong['entry_price']) / wrong['entry_price'] * 100
        print(f"  If calculated as long position: {wrong_calc:.4f}%")