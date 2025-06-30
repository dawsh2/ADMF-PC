#\!/usr/bin/env python3
"""
Analyze filter effectiveness by comparing signal counts across strategies.
"""

import json
import pandas as pd
from pathlib import Path

# Load metadata
metadata_path = Path("config/keltner/results/latest/metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract strategy info
strategies = []
for name, comp in metadata['components'].items():
    if name.startswith('SPY_5m_compiled_strategy_'):
        strategy_num = int(name.split('_')[-1])
        strategies.append({
            'strategy': strategy_num,
            'signals': comp.get('signal_changes', 0),
            'compression': comp.get('compression_ratio', 0)
        })

# Convert to DataFrame and sort
df = pd.DataFrame(strategies)
df = df.sort_values('signals')

print("=== Filter Effectiveness Analysis ===\n")

# Group by signal count to identify filter groups
signal_groups = df.groupby('signals').agg({
    'strategy': ['count', 'min', 'max', list]
}).reset_index()

print("Signal Count Groups:")
print(f"{'Signals':<10} {'Count':<8} {'Strategy Range':<20} {'Example Strategies'}")
print("-" * 70)

for _, row in signal_groups.iterrows():
    signals = int(row['signals'])
    count = int(row['strategy']['count'])
    min_strat = int(row['strategy']['min'])
    max_strat = int(row['strategy']['max'])
    examples = row['strategy']['list'][:3]  # First 3 examples
    
    print(f"{signals:<10} {count:<8} {min_strat}-{max_strat:<17} {str(examples)}")

# Identify baseline (most signals)
baseline_signals = df['signals'].max()
print(f"\nBaseline (unfiltered): {baseline_signals} signals")

# Calculate filter effectiveness
print("\nFilter Effectiveness (% reduction from baseline):")
print(f"{'Signals':<10} {'Reduction %':<15} {'Likely Filter Type'}")
print("-" * 50)

for signals in sorted(signal_groups['signals'].unique()):
    if signals < baseline_signals:
        reduction = (1 - signals / baseline_signals) * 100
        
        # Guess filter type based on reduction
        if reduction > 98:
            filter_type = "Heavy filtering (multi-condition)"
        elif reduction > 95:
            filter_type = "Strong filtering (volatility/time)"
        elif reduction > 90:
            filter_type = "Moderate filtering (RSI/volume)"
        elif reduction > 80:
            filter_type = "Light filtering"
        else:
            filter_type = "Minimal filtering"
        
        print(f"{signals:<10} {reduction:>6.1f}%         {filter_type}")

# Expected filters from config
print("\n=== Expected Filters from Config ===")
print("""
Based on the config, we expect:
1. Baseline (no filter): ~3,200+ signals
2. RSI filters: Moderate reduction
3. Volume filters: Moderate reduction  
4. Volatility regime filter: Strong reduction
5. VWAP positioning: Moderate reduction
6. Time of day filter: ~20-30% reduction
7. Combined filters: Very strong reduction
8. Master regime filter: 95%+ reduction (47 signals)
9. Long-only variant: ~50% reduction

The 47-signal strategies are likely our regime-based filters working correctly!
""")