#!/usr/bin/env python3
"""
Check if features are being inferred from filters.
"""

# The RSI filter uses "rsi(14) < threshold"
# This requires the RSI feature to be calculated

print("🔍 Checking Feature Inference:\n")

print("Filter expressions in config:")
print("  - RSI: 'rsi(14) < threshold'")
print("  - Volume: 'volume > volume_sma_20 * multiplier'")
print("  - Volatility: 'atr(14) > atr_sma(50) * threshold'")

print("\nRequired features:")
print("  - rsi_14 (for RSI filter)")
print("  - volume (raw volume data)")
print("  - volume_sma_20 (20-period volume SMA)")
print("  - atr_14 (14-period ATR)")
print("  - atr_sma_50 (50-period SMA of ATR)")

print("\nPossible issues:")
print("1. Features might not be auto-inferred from filter expressions")
print("2. RSI might not be calculated, causing filter to fail silently")
print("3. Filter syntax might need exact feature names (rsi_14 vs rsi(14))")

print("\n💡 Solution:")
print("The config mentions: '# Features are automatically inferred from filter usage'")
print("But this might require explicit feature configuration.")

# Check if we can see the features in metadata
from pathlib import Path
import json

metadata_path = Path("config/keltner/results/latest/metadata.json")
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Check if features are mentioned
    config_data = metadata.get('config', {})
    features = config_data.get('features', {})
    feature_configs = config_data.get('feature_configs', {})
    
    print(f"\n📊 Features in metadata:")
    print(f"  features: {features}")
    print(f"  feature_configs: {feature_configs}")
    
    # Check execution section
    execution = config_data.get('execution', {})
    print(f"\n📊 Execution config:")
    print(f"  {json.dumps(execution, indent=2)}")