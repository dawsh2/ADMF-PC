#!/usr/bin/env python3
"""Debug why volume strategies aren't generating signals."""

import subprocess
import sys

# Run a quick test focusing on volume strategies
cmd = ['python', 'main.py', '--config', 'config/expansive_grid_search.yaml', '--signal-generation', '--bars', '200']
result = subprocess.run(cmd, capture_output=True, text=True)

print("=== VOLUME STRATEGY ANALYSIS ===")

# Count signals by volume strategies
volume_strategies = ['obv_trend', 'vwap_deviation', 'accumulation_distribution', 'chaikin_money_flow', 'mfi_bands']

for strategy in volume_strategies:
    signals = []
    for line in result.stdout.split('\n'):
        if '📡 SIGNAL:' in line and strategy in line:
            signals.append(line)
    
    print(f"\n{strategy}: {len(signals)} signals")
    if signals:
        print(f"  Sample: {signals[0]}")
    else:
        print(f"  ❌ No signals found")

# Look for any volume-related errors
print(f"\n=== ERROR ANALYSIS ===")
volume_keywords = ['obv', 'vwap', 'volume', 'ad_line', 'cmf']
errors = []

for line in result.stderr.split('\n'):
    if any(keyword in line.lower() for keyword in volume_keywords) and ('error' in line.lower() or 'warning' in line.lower()):
        errors.append(line)

if errors:
    print("Volume-related errors/warnings:")
    for error in errors[:5]:  # Show first 5
        print(f"  {error}")
else:
    print("No volume-related errors found")

# Check if features are being configured
print(f"\n=== FEATURE CONFIGURATION ===")
feature_lines = []
for line in result.stderr.split('\n'):
    if 'feature' in line.lower() and ('obv' in line.lower() or 'vwap' in line.lower() or 'ad' in line.lower()):
        feature_lines.append(line)

if feature_lines:
    print("Volume feature configuration:")
    for line in feature_lines[:3]:  # Show first 3
        print(f"  {line}")
else:
    print("No volume feature configuration found in logs")

# Check for component readiness
print(f"\n=== READINESS ANALYSIS ===")
readiness_lines = []
for line in result.stderr.split('\n'):
    if 'ready' in line.lower() and any(vs in line for vs in ['obv_trend', 'vwap_deviation']):
        readiness_lines.append(line)

if readiness_lines:
    print("Volume strategy readiness:")
    for line in readiness_lines:
        print(f"  {line}")
else:
    print("No readiness info found for volume strategies")