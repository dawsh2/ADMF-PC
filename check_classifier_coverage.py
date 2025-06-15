#!/usr/bin/env python3
"""
Check classifier coverage and overlap
"""
import pandas as pd
from pathlib import Path

workspace = Path("workspaces/expansive_grid_search_0397bd70")
classifiers_dir = workspace / "traces" / "SPY_1m" / "classifiers"

print("CLASSIFIER COVERAGE ANALYSIS")
print("="*80)

total_bars = 80000
classifier_stats = {}

for classifier_type_dir in classifiers_dir.iterdir():
    if not classifier_type_dir.is_dir():
        continue
    
    classifier_type = classifier_type_dir.name
    all_timestamps = set()
    regime_counts = {}
    
    # Load all files for this classifier type
    for file_path in classifier_type_dir.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        
        # Add timestamps to set
        all_timestamps.update(df['ts'].unique())
        
        # Count regimes
        for regime in df['val'].value_counts().items():
            regime_name, count = regime
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + count
    
    classifier_stats[classifier_type] = {
        'unique_bars': len(all_timestamps),
        'coverage_pct': len(all_timestamps) / total_bars * 100,
        'total_classifications': sum(regime_counts.values()),
        'regimes': regime_counts
    }

# Print results
for classifier, stats in sorted(classifier_stats.items()):
    print(f"\n{classifier}:")
    print(f"  Unique bars classified: {stats['unique_bars']:,} ({stats['coverage_pct']:.1f}%)")
    print(f"  Total classifications: {stats['total_classifications']:,}")
    
    if stats['total_classifications'] > stats['unique_bars']:
        print(f"  ⚠️  OVERLAPPING: {stats['total_classifications'] - stats['unique_bars']:,} duplicate classifications!")
    
    print(f"  Regime distribution:")
    for regime, count in sorted(stats['regimes'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_classifications'] * 100
        print(f"    {regime:<20} {count:>7,} ({pct:>5.1f}%)")

print("\n" + "="*80)
print("SUMMARY:")
print(f"Total bars in dataset: {total_bars:,}")
print(f"Average classifier coverage: {sum(s['coverage_pct'] for s in classifier_stats.values()) / len(classifier_stats):.1f}%")