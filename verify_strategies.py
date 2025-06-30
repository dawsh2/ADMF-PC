#!/usr/bin/env python3
"""Verify we're comparing the same strategies between universal analysis and execution."""

import pandas as pd
from pathlib import Path

# Load strategy indices from both runs
universal_run = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448")
execution_run = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812")

# Load strategy index from universal analysis
universal_strategies = pd.read_parquet(universal_run / "strategy_index.parquet")
print("Universal Analysis Strategies:")
print(f"Total strategies: {len(universal_strategies)}")
print("\nFirst 5 strategies:")
print(universal_strategies[['strategy_hash', 'period', 'std_dev']].head())

# Find the top strategy (hash: 5edc43651004)
top_strategy = universal_strategies[universal_strategies['strategy_hash'].str.startswith('5edc4365')]
if len(top_strategy) > 0:
    print(f"\nTop strategy from universal analysis:")
    print(f"Hash: {top_strategy.iloc[0]['strategy_hash']}")
    print(f"Period: {top_strategy.iloc[0]['period']}")
    print(f"Std dev: {top_strategy.iloc[0]['std_dev']}")

# Check execution configuration
import yaml
config_path = Path("/Users/daws/ADMF-PC/config/bollinger/test.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
    
print("\nExecution Configuration:")
print(f"Strategy type: {list(config['strategy'][0].keys())[0]}")
params = config['strategy'][0]['bollinger_bands']
print(f"Period: {params['period']}")
print(f"Std dev: {params['std_dev']}")
print(f"\nRisk parameters:")
risk = config['strategy'][0]['risk']
print(f"Stop loss: {risk['stop_loss']} ({risk['stop_loss']*100}%)")
print(f"Take profit: {risk['take_profit']} ({risk['take_profit']*100}%)")

# Check if they match
print("\n" + "="*50)
print("COMPARISON:")
print(f"Periods match: {top_strategy.iloc[0]['period'] == params['period']}")
print(f"Std devs match: {top_strategy.iloc[0]['std_dev'] == params['std_dev']}")
print(f"\nStop/target match optimization: {risk['stop_loss']*100:.3f}% / {risk['take_profit']*100:.3f}%")
print("Expected from optimization: 0.075% / 0.100%")

# Check execution signals
execution_signals = list((execution_run / "traces/signals").rglob("*.parquet"))
if execution_signals:
    signals = pd.read_parquet(execution_signals[0])
    print(f"\nExecution has {len(signals)} signal records")