#!/usr/bin/env python3
"""
Verify debug test results to see if RSI filter is working.
"""

import pandas as pd
from pathlib import Path

# Find results in debug directory
results_dir = Path("config/debug/results")
if not results_dir.exists():
    print("❌ No results directory found. Run the debug config first:")
    print("   python main.py --config config/debug/config.yaml --signal-generation --bars 100")
    exit(1)

# Get latest results
result_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != 'latest']
if result_dirs:
    latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
else:
    print("❌ No result directories found")
    exit(1)

print(f"📁 Analyzing debug results in: {latest_dir}\n")

traces_dir = latest_dir / "traces" / "keltner_bands"
if not traces_dir.exists():
    print(f"❌ No traces found in {traces_dir}")
    exit(1)

# Expected strategies from debug config:
# 0: No filter (baseline)
# 1: RSI < 50 (single threshold)
# 2-4: RSI < [30, 50, 70] (expanded)
# 5: Direct expression "signal == 0 or rsi_14 < 50"
# 6: Volume > 1.2x
# 7: Combined RSI + Volume

strategy_files = list(traces_dir.glob("*.parquet"))
print(f"📊 Found {len(strategy_files)} strategy files\n")

# Analyze each strategy
results = []
for i in range(8):  # We expect 8 strategies
    file_path = traces_dir / f"SPY_5m_compiled_strategy_{i}.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
        signal_changes = len(df)
        buy_signals = (df['val'] > 0).sum()
        sell_signals = (df['val'] < 0).sum()
        flat_signals = (df['val'] == 0).sum()
        
        results.append({
            'strategy': i,
            'changes': signal_changes,
            'buys': buy_signals,
            'sells': sell_signals,
            'flats': flat_signals
        })

# Display results
print("📈 Strategy Analysis:")
print("-" * 60)
print("ID | Filter Type              | Changes | Buy | Sell | Flat")
print("-" * 60)

filter_names = [
    "No filter (baseline)",
    "RSI < 50",
    "RSI < 30",
    "RSI < 50",
    "RSI < 70",
    "Direct: rsi_14 < 50",
    "Volume > 1.2x",
    "RSI < 50 AND Volume > 1.2x"
]

for i, result in enumerate(results):
    if i < len(filter_names):
        name = filter_names[i]
    else:
        name = f"Strategy {i}"
    
    print(f"{result['strategy']:2d} | {name:24s} | {result['changes']:7d} | {result['buys']:3d} | {result['sells']:4d} | {result['flats']:4d}")

# Calculate reductions
if results:
    baseline = results[0]['changes']
    print("\n📊 Filter Effectiveness (vs baseline):")
    print("-" * 40)
    
    for i in range(1, len(results)):
        if i < len(filter_names):
            name = filter_names[i]
        else:
            name = f"Strategy {i}"
        
        reduction = (1 - results[i]['changes'] / baseline) * 100
        print(f"{name:30s}: {reduction:5.1f}% reduction")

print("\n✅ Analysis complete!")
print("\n💡 Key insights:")
print("- If RSI filters show 0% reduction, the feature naming issue persists")
print("- If RSI filters show reduction, the fix worked!")
print("- Volume filter should show ~35-65% reduction as a baseline")