#!/usr/bin/env python3
"""
Test what features each strategy actually needs
"""
import importlib
from collections import defaultdict

strategy_modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.structure',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
]

all_features = defaultdict(set)

print("Analyzing feature requirements for each strategy...\n")

for module_path in strategy_modules:
    try:
        module = importlib.import_module(module_path)
        strategies = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
        
        for strategy_name in strategies:
            func = getattr(module, strategy_name)
            if hasattr(func, '_strategy_metadata'):
                metadata = func._strategy_metadata
                feature_config = metadata.get('feature_config', {})
                
                print(f"{strategy_name}:")
                for feature_type, config in feature_config.items():
                    params = config.get('params', [])
                    defaults = config.get('defaults', {})
                    print(f"  - {feature_type}: params={params}, defaults={defaults}")
                    
                    # Track what features are needed
                    for param in params:
                        param_value = defaults.get(param, 'UNKNOWN')
                        if isinstance(param_value, (int, float)):
                            all_features[feature_type].add(param_value)
                        
    except Exception as e:
        print(f"Error importing {module_path}: {e}")

print("\n=== SUMMARY OF REQUIRED FEATURES ===")
for feature_type, periods in sorted(all_features.items()):
    print(f"{feature_type}: {sorted(periods)}")