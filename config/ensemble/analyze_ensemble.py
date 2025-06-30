#!/usr/bin/env python3
"""
Analyze ensemble configuration and explain why no signals were generated
"""

import yaml
from pathlib import Path

def analyze_ensemble_config():
    """Analyze the ensemble configuration."""
    
    # Load the config
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Ensemble Configuration Analysis ===")
    print(f"Name: {config.get('name', 'unnamed')}")
    print(f"Data: {config.get('data', 'unknown')}")
    print(f"\nStrategies:")
    
    strategies = config.get('strategy', [])
    for i, strategy in enumerate(strategies):
        print(f"\n{i+1}. Strategy Configuration:")
        for strategy_type, params in strategy.items():
            print(f"   Type: {strategy_type}")
            print(f"   Parameters: {params}")
    
    print("\n=== Analysis ===")
    print("\nThis configuration defines individual strategies, not an ensemble.")
    print("In ADMF-PC, when multiple strategies are listed like this:")
    print("- Each strategy runs independently")
    print("- Each generates its own signals")
    print("- They are NOT combined into an ensemble")
    
    print("\nTo create a true ensemble, you would need:")
    print("1. A specific ensemble strategy type (e.g., 'two_layer_ensemble')")
    print("2. Sub-strategies defined within the ensemble configuration")
    print("3. Combination rules (voting, weighting, etc.)")
    
    print("\n=== Example True Ensemble Config ===")
    example_ensemble = """
strategy:
  - type: two_layer_ensemble
    name: my_ensemble
    params:
      baseline_strategies:
        - type: keltner_bands
          params: {period: 26, multiplier: 3.0}
          weight: 0.5
        - type: bollinger_bands  
          params: {period: 11, std_dev: 2.0}
          weight: 0.5
      regime_strategies:
        - regime: trending
          strategies:
            - type: ma_crossover
              params: {fast_period: 10, slow_period: 20}
      combination_method: weighted_vote
      threshold: 0.5
"""
    print(example_ensemble)
    
    print("\n=== Why No Signals? ===")
    print("Possible reasons:")
    print("1. The strategies might not have generated any signals with these parameters")
    print("2. Keltner (period=26, mult=3.0) - very wide bands, few signals")
    print("3. Bollinger (period=11, std=2.0) - standard bands")
    print("4. Both might be too conservative for the data period")
    
    # Check metadata for more clues
    metadata_path = Path("results/latest/metadata.json")
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nMetadata shows:")
        print(f"- Total bars processed: {metadata.get('total_bars', 0)}")
        print(f"- Total signals: {metadata.get('total_signals', 0)}")
        print(f"- Components: {len(metadata.get('components', {}))}")
        
        if metadata.get('total_bars', 0) > 0 and metadata.get('total_signals', 0) == 0:
            print("\nThe strategies processed data but generated no signals.")
            print("This suggests the parameters are too conservative.")

if __name__ == "__main__":
    analyze_ensemble_config()