#!/usr/bin/env python3
"""
Analyze the mismatch between how features are generated and how strategies request them.
"""

import json
from src.core.components.registry import get_global_registry

# Get the strategy registry
registry = get_global_registry()
STRATEGY_REGISTRY = {name: registry.get_class(name) for name in registry.list_all() 
                     if registry.get(name) and 'strategy' in str(registry.get(name).tags)}

def analyze_feature_naming():
    """Analyze how features are named when generated vs requested."""
    
    print("=== FEATURE NAMING ANALYSIS ===\n")
    
    # Example: stochastic crossover
    strat = STRATEGY_REGISTRY.get('stochastic_crossover')
    if strat and hasattr(strat, '_strategy_metadata'):
        metadata = strat._strategy_metadata
        feature_config = metadata.get('feature_config', {})
        
        print("STOCHASTIC CROSSOVER EXAMPLE:")
        print(f"Feature config: {json.dumps(feature_config, indent=2)}")
        
        # Show what the strategy looks for
        print("\nStrategy looks for features with these names:")
        k_period = 14  # default
        d_period = 3   # default
        print(f"  - stochastic_{k_period}_{d_period}_k")
        print(f"  - stochastic_{k_period}_{d_period}_d")
        
        print("\nBut feature hub would generate:")
        print("  - If feature name is 'stochastic_14_3':")
        print("    - stochastic_14_3_k")
        print("    - stochastic_14_3_d")
        print("  - If feature name is just 'stochastic':")
        print("    - stochastic_k")
        print("    - stochastic_d")
    
    print("\n" + "="*50 + "\n")
    
    # Analyze all strategies
    mismatches = []
    
    for strat_name, strat_func in STRATEGY_REGISTRY.items():
        if hasattr(strat_func, '_strategy_metadata'):
            metadata = strat_func._strategy_metadata
            feature_config = metadata.get('feature_config', {})
            
            for feature_type, config in feature_config.items():
                params = config.get('params', [])
                defaults = config.get('defaults', {})
                
                # Check specific problematic patterns
                if feature_type == 'stochastic':
                    k_period = defaults.get('k_period', 14)
                    d_period = defaults.get('d_period', 3)
                    expected = f"stochastic_{k_period}_{d_period}_k"
                    mismatches.append({
                        'strategy': strat_name,
                        'feature_type': feature_type,
                        'expected_pattern': expected,
                        'issue': 'Feature name in config must include parameters'
                    })
                
                elif feature_type == 'macd':
                    fast = defaults.get('fast_ema', 12)
                    slow = defaults.get('slow_ema', 26)
                    signal = defaults.get('signal_ema', 9)
                    expected = f"macd_{fast}_{slow}_{signal}_macd"
                    mismatches.append({
                        'strategy': strat_name,
                        'feature_type': feature_type,
                        'expected_pattern': expected,
                        'issue': 'Feature name in config must include all three parameters'
                    })
                
                elif feature_type == 'stochastic_rsi':
                    rsi_period = defaults.get('rsi_period', 14)
                    stoch_period = defaults.get('stoch_period', 14)
                    expected = f"stochastic_rsi_{rsi_period}_{stoch_period}_k"
                    mismatches.append({
                        'strategy': strat_name,
                        'feature_type': feature_type,
                        'expected_pattern': expected,
                        'issue': 'Feature name in config must include both periods'
                    })
    
    print("IDENTIFIED MISMATCHES:")
    for mismatch in mismatches[:10]:  # Show first 10
        print(f"\nStrategy: {mismatch['strategy']}")
        print(f"Feature type: {mismatch['feature_type']}")
        print(f"Expected pattern: {mismatch['expected_pattern']}")
        print(f"Issue: {mismatch['issue']}")
    
    print(f"\nTotal potential mismatches: {len(mismatches)}")
    
    # Show how features should be configured
    print("\n" + "="*50)
    print("CORRECT FEATURE CONFIGURATION PATTERN:")
    print("""
For features to work correctly, the feature config name must match 
what the strategy looks for. Examples:

Instead of:
  features:
    stochastic:
      feature: stochastic
      k_period: 14
      d_period: 3

Use:
  features:
    stochastic_14_3:  # Name includes parameters!
      feature: stochastic
      k_period: 14
      d_period: 3

This will generate features named:
  - stochastic_14_3_k
  - stochastic_14_3_d

Which matches what the strategy looks for!
""")

if __name__ == '__main__':
    analyze_feature_naming()