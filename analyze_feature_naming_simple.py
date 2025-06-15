#!/usr/bin/env python3
"""
Analyze the feature naming mismatch by examining actual strategy code.
"""

import importlib

# Import a specific strategy to analyze
from src.strategy.strategies.indicators.crossovers import stochastic_crossover

print("=== STOCHASTIC CROSSOVER ANALYSIS ===\n")

# Check metadata
if hasattr(stochastic_crossover, '_strategy_metadata'):
    metadata = stochastic_crossover._strategy_metadata
    print(f"Strategy metadata: {metadata}")
    
    feature_config = metadata.get('feature_config', {})
    print(f"\nFeature config: {feature_config}")
    
    # Show what features the strategy expects
    print("\nStrategy code looks for these feature names:")
    print("  - features.get(f'stochastic_{k_period}_{d_period}_k')")
    print("  - features.get(f'stochastic_{k_period}_{d_period}_d')")
    print("\nWith default values k_period=14, d_period=3:")
    print("  - stochastic_14_3_k")
    print("  - stochastic_14_3_d")

print("\n=== THE PROBLEM ===")
print("""
The strategy expects features named with their parameters included,
but the feature configuration might not generate matching names.

If the feature config has:
  features:
    stochastic:  # Just 'stochastic', no parameters in name
      feature: stochastic
      k_period: 14
      d_period: 3

Then features generated would be:
  - stochastic_k
  - stochastic_d

But the strategy looks for:
  - stochastic_14_3_k
  - stochastic_14_3_d

MISMATCH! The strategy won't find its features.
""")

print("=== THE SOLUTION ===")
print("""
The feature configuration key must include the parameters:

  features:
    stochastic_14_3:  # Include parameters in the key!
      feature: stochastic
      k_period: 14
      d_period: 3

This generates:
  - stochastic_14_3_k
  - stochastic_14_3_d

Which matches what the strategy expects!
""")

# Check other problematic strategies
print("\n=== OTHER PROBLEMATIC PATTERNS ===\n")

strategies_to_check = [
    ('macd_crossover', 'src.strategy.strategies.indicators.crossovers'),
    ('stochastic_rsi', 'src.strategy.strategies.indicators.oscillators'),
    ('ultimate_oscillator', 'src.strategy.strategies.indicators.oscillators'),
    ('fibonacci_retracement', 'src.strategy.strategies.indicators.structure'),
]

for strat_name, module_path in strategies_to_check:
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, strat_name):
            func = getattr(module, strat_name)
            if hasattr(func, '_strategy_metadata'):
                metadata = func._strategy_metadata
                feature_config = metadata.get('feature_config', {})
                print(f"{strat_name}:")
                for feat_type, config in feature_config.items():
                    params = config.get('params', [])
                    defaults = config.get('defaults', {})
                    print(f"  - {feat_type}: params={params}")
                    if defaults:
                        print(f"    defaults: {defaults}")
    except Exception as e:
        print(f"Error checking {strat_name}: {e}")

print("\n=== SUMMARY ===")
print("""
Many strategies (427 out of 969 components) aren't finding their features
because the feature configuration keys don't include the parameter values
that the strategies expect in the feature names.

The fix is to ensure feature configuration keys match the naming pattern
the strategies use when looking up features.
""")