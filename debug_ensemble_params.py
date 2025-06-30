#!/usr/bin/env python3
"""Debug why ensemble parameters aren't being passed."""

import yaml
import json

print("DEBUGGING ENSEMBLE PARAMETER PASSING")
print("=" * 50)

# 1. Check ensemble config
print("\n1. Ensemble config (config/ensemble/config.yaml):")
with open('config/ensemble/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"   Full config: {config}")
print(f"   Strategy: {config['strategy']}")
print(f"   First strategy: {config['strategy'][0]}")
print(f"   Bollinger params: {config['strategy'][0]['bollinger_bands']}")

# 2. Check latest metadata
print("\n2. Latest metadata (what was actually used):")
with open('config/ensemble/results/20250623_144646/metadata.json', 'r') as f:
    metadata = json.load(f)

strategies = metadata['strategy_metadata']['strategies']
for name, strat in strategies.items():
    print(f"   Strategy '{name}':")
    print(f"     Type: {strat['type']}")
    print(f"     Params: {strat['params']}")
    print(f"     ^^^ EMPTY! This is the problem!")

# 3. Check parameter sweep to compare
print("\n3. Parameter sweep metadata for comparison:")
sweep_meta_path = 'config/bollinger/results/20250623_062931/metadata.json'
try:
    with open(sweep_meta_path, 'r') as f:
        sweep_meta = json.load(f)
    
    # Look at a few strategies
    sweep_strats = sweep_meta.get('strategy_metadata', {}).get('strategies', {})
    count = 0
    for name, strat in sweep_strats.items():
        if count < 3:  # Just show first 3
            print(f"   Strategy '{name}':")
            print(f"     Type: {strat['type']}")
            print(f"     Params: {strat['params']}")
            count += 1
except:
    print("   Could not load parameter sweep metadata")

print("\n4. DIAGNOSIS:")
print("-" * 40)
print("The ensemble is NOT passing parameters to the strategy!")
print("Even though config.yaml specifies {period: 15, std_dev: 3.0},")
print("the metadata shows params: {} (empty)")
print("\nThis means the strategy is using defaults:")
print("- period: 20 (not 15)")
print("- std_dev: 2.0 (not 3.0)")
print("- exit_threshold: 0.001")

print("\n5. NEXT STEPS:")
print("-" * 40)
print("Need to debug the compiler/coordinator to see why")
print("parameters aren't being passed from config to strategy.")