#!/usr/bin/env python3
"""Diagnose why Keltner Bands strategy isn't running."""

import yaml
import json
from pathlib import Path

print("=== KELTNER BANDS DIAGNOSTIC ===\n")

# Check config file
config_path = Path("config/indicators/volatility/test_keltner_bands_working.yaml")
if config_path.exists():
    print("✓ Config file exists")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"  Config name: {config.get('name')}")
    print(f"  Mode: {config.get('mode')}")
    print(f"  Strategy: {list(config.get('strategy', {}).keys())}")
else:
    print("✗ Config file not found!")

# Check strategy implementation
print("\n--- Strategy Implementation ---")
strategy_file = Path("src/strategy/strategies/indicators/volatility.py")
if strategy_file.exists():
    with open(strategy_file, 'r') as f:
        content = f.read()
    if 'def keltner_bands' in content:
        print("✓ keltner_bands function found")
        # Find the decorator
        import re
        decorator_match = re.search(r'@strategy\([^)]+name=[\'"]keltner_bands[\'"][^)]*\)', content, re.DOTALL)
        if decorator_match:
            print("✓ @strategy decorator found")
    else:
        print("✗ keltner_bands function not found!")
else:
    print("✗ Strategy file not found!")

# Check feature implementation
print("\n--- Feature Implementation ---")
feature_file = Path("src/strategy/components/features/indicators/volatility.py")
if feature_file.exists():
    with open(feature_file, 'r') as f:
        content = f.read()
    if 'class KeltnerChannel' in content:
        print("✓ KeltnerChannel class found")
    if 'keltner_channel' in content.lower():
        print("✓ keltner_channel registered in VOLATILITY_FEATURES")
else:
    print("✗ Feature file not found!")

# Check recent workspaces
print("\n--- Recent Workspaces ---")
workspaces_dir = Path("workspaces")
if workspaces_dir.exists():
    # Get 5 most recent signal_generation workspaces
    sg_workspaces = sorted([d for d in workspaces_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('signal_generation_')],
                          key=lambda x: x.stat().st_mtime, reverse=True)[:5]
    
    for ws in sg_workspaces:
        metadata_file = ws / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            total = metadata.get('strategy_metadata', {}).get('summary', {}).get('total_strategies', 0)
            signals = metadata.get('total_signals', 0)
            print(f"  {ws.name}: {total} strategies, {signals} signals")

# Check if we can import the strategy
print("\n--- Import Test ---")
try:
    from src.strategy.strategies.indicators.volatility import keltner_bands
    print("✓ Successfully imported keltner_bands")
    print(f"  Function: {keltner_bands}")
    print(f"  Has metadata: {hasattr(keltner_bands, '_strategy_metadata')}")
except Exception as e:
    print(f"✗ Import failed: {e}")

# Check discovery
print("\n--- Strategy Discovery ---")
try:
    from src.core.components.discovery import get_discovered_strategies
    strategies = get_discovered_strategies()
    if 'keltner_bands' in strategies:
        print("✓ keltner_bands found in discovered strategies")
        print(f"  Metadata: {strategies['keltner_bands']}")
    else:
        print(f"✗ keltner_bands not in discovered strategies")
        print(f"  Available volatility strategies: {[s for s in strategies.keys() if 'keltner' in s or 'bands' in s]}")
except Exception as e:
    print(f"✗ Discovery failed: {e}")

print("\n=== END DIAGNOSTIC ===")