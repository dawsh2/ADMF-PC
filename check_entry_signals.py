#!/usr/bin/env python3
"""Check entry signal values in opens."""

import pandas as pd
import json
from pathlib import Path

results_dir = Path("config/bollinger/results/latest")
opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/positions_open.parquet")

# Parse metadata
entry_signals = []
for i in range(len(opens)):
    if isinstance(opens.iloc[i]['metadata'], str):
        try:
            meta = json.loads(opens.iloc[i]['metadata'])
            # Check nested metadata
            if 'metadata' in meta and 'entry_signal' in meta['metadata']:
                entry_signals.append(meta['metadata']['entry_signal'])
            elif 'entry_signal' in meta:
                entry_signals.append(meta['entry_signal'])
            else:
                entry_signals.append(None)
        except:
            entry_signals.append(None)

print("=== Entry Signals in Position Opens ===")
print(f"\nTotal opens: {len(opens)}")
print(f"Entry signals found: {sum(1 for s in entry_signals if s is not None)}")

# Count values
from collections import Counter
signal_counts = Counter(entry_signals)
print(f"\nEntry signal distribution:")
for signal, count in sorted(signal_counts.items()):
    print(f"  {signal}: {count}")

# Check quantity to see if we have shorts
quantities = []
for i in range(len(opens)):
    if isinstance(opens.iloc[i]['metadata'], str):
        try:
            meta = json.loads(opens.iloc[i]['metadata'])
            quantities.append(meta.get('quantity', 0))
        except:
            quantities.append(0)

print(f"\nQuantity distribution:")
qty_counts = Counter(quantities)
for qty, count in sorted(qty_counts.items()):
    if qty != 0:
        print(f"  {qty}: {count} ({'LONG' if qty > 0 else 'SHORT'})")