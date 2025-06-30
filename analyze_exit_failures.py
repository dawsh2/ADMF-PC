"""Analyze why some exits are not at exact prices."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

# Analyze stop losses that are NOT at exact -0.075%
print("=== ANALYZING STOP LOSS FAILURES ===\n")

stop_losses = []
for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type') == 'stop_loss':
            # Find the previous entry
            for j in range(idx-1, -1, -1):
                prev_row = fills_df.iloc[j]
                prev_meta = prev_row['metadata']
                if isinstance(prev_meta, dict):
                    prev_nested = prev_meta.get('metadata', {})
                    if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                        entry_price = float(prev_meta.get('price', 0))
                        exit_price = float(metadata.get('price', 0))
                        if entry_price > 0:
                            return_pct = (exit_price - entry_price) / entry_price * 100
                            is_exact = abs(return_pct - (-0.075)) < 0.001
                            
                            # For short positions, the math is reversed
                            short_side = metadata.get('side') == 'BUY'  # Exit BUY means position was SHORT
                            if short_side:
                                return_pct = -return_pct
                                is_exact = abs(return_pct - (-0.075)) < 0.001
                            
                            stop_losses.append({
                                'idx': idx,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'return_pct': return_pct,
                                'is_exact': is_exact,
                                'is_short': short_side,
                                'exit_reason': nested.get('exit_reason', ''),
                                'order_id': metadata.get('order_id', ''),
                                'timestamp': row.get('executed_at', row.get('timestamp', ''))
                            })
                            break

# Analyze failures
failures = [sl for sl in stop_losses if not sl['is_exact']]
print(f"Total stop losses: {len(stop_losses)}")
print(f"Failures (not exact): {len(failures)}")
print(f"Success rate: {(len(stop_losses) - len(failures)) / len(stop_losses) * 100:.1f}%\n")

# Show examples of failures
print("Examples of FAILED stop losses (not at -0.075%):")
for i, fail in enumerate(failures[:5]):
    print(f"\nFailure {i+1}:")
    print(f"  Timestamp: {fail['timestamp']}")
    print(f"  Entry: ${fail['entry_price']:.4f}")
    print(f"  Exit:  ${fail['exit_price']:.4f}")
    print(f"  Return: {fail['return_pct']:.4f}% (expected: -0.0750%)")
    print(f"  Short position: {fail['is_short']}")
    print(f"  Exit reason: {fail['exit_reason']}")
    print(f"  Order ID: {fail['order_id']}")

# Check for pattern in failures
print("\n=== FAILURE ANALYSIS ===")

# Are failures mostly short positions?
short_failures = [f for f in failures if f['is_short']]
long_failures = [f for f in failures if not f['is_short']]
print(f"\nShort position failures: {len(short_failures)}/{len(failures)} ({len(short_failures)/len(failures)*100:.1f}%)")
print(f"Long position failures: {len(long_failures)}/{len(failures)} ({len(long_failures)/len(failures)*100:.1f}%)")

# Check if failures have inverted returns (e.g., +0.075% instead of -0.075%)
inverted_failures = [f for f in failures if abs(f['return_pct'] - 0.075) < 0.01]
print(f"\nInverted returns (+0.075% instead of -0.075%): {len(inverted_failures)}")

if inverted_failures:
    print("\nExamples of inverted returns:")
    for i, inv in enumerate(inverted_failures[:3]):
        print(f"  Entry: ${inv['entry_price']:.4f}, Exit: ${inv['exit_price']:.4f}, Return: {inv['return_pct']:.4f}%, Short: {inv['is_short']}")