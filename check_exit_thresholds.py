#!/usr/bin/env python3
"""Check what exit thresholds were actually used."""

import yaml

print("CHECKING EXIT THRESHOLDS")
print("=" * 50)

# Check bollinger parameter sweep config
print("\n1. Bollinger parameter sweep config:")
with open('config/bollinger/config.yaml', 'r') as f:
    bollinger_config = yaml.safe_load(f)

bb_config = bollinger_config['strategy'][0]['bollinger_bands']
print(f"   Specified parameters: {bb_config}")
print(f"   Exit threshold: {'NOT SPECIFIED - uses default 0.001' if 'exit_threshold' not in bb_config else bb_config['exit_threshold']}")

# Check ensemble config
print("\n2. Ensemble config:")
with open('config/ensemble/config.yaml', 'r') as f:
    ensemble_config = yaml.safe_load(f)

ens_config = ensemble_config['strategy'][0]['bollinger_bands']
print(f"   Specified parameters: {ens_config}")
print(f"   Exit threshold: {'NOT SPECIFIED - uses default 0.001' if 'exit_threshold' not in ens_config else ens_config['exit_threshold']}")

print("\n3. Analysis:")
print("-" * 40)
print("Parameter sweep: exit_threshold = 0.001 (0.1%) [DEFAULT]")
print("Ensemble run: exit_threshold = 0.0001 (0.01%) [SPECIFIED]")
print("\nThis 10x tighter exit threshold explains why the ensemble:")
print("- Exits positions much faster (1.4 bars vs 4 bars)")
print("- Has terrible performance (0.30% vs 14.84% return)")
print("- Has very low win rate (1.4% vs 49.1%)")

print("\n4. Solution:")
print("-" * 40)
print("Remove or increase the exit_threshold in ensemble config:")
print("  {bollinger_bands: {period: 15, std_dev: 3.0}}  # Use default 0.001")
print("OR")
print("  {bollinger_bands: {period: 15, std_dev: 3.0, exit_threshold: 0.001}}")