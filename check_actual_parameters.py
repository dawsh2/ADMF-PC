#!/usr/bin/env python3
"""
Check what parameters are actually being used by examining the feature names.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.compiler import StrategyCompiler
from src.core.coordinator.config.clean_syntax_parser import CleanSyntaxParser
import yaml

def check_parameters():
    """Check what parameters the strategy is actually using."""
    
    print("Checking Actual Parameters Used")
    print("=" * 50)
    
    # Load and parse config
    with open('config/ensemble/config.yaml', 'r') as f:
        raw_config = yaml.safe_load(f)
    
    print("\n1. Raw config:")
    print(f"   {raw_config}")
    
    # Parse with clean syntax
    parser = CleanSyntaxParser()
    parsed_config = parser.parse_config(raw_config)
    
    print("\n2. After clean syntax parsing:")
    print(f"   {parsed_config}")
    
    # Compile strategies
    compiler = StrategyCompiler()
    compiled_strategies = compiler.compile_strategies(parsed_config)
    
    print(f"\n3. Compiled {len(compiled_strategies)} strategies")
    
    for i, strategy in enumerate(compiled_strategies):
        print(f"\n   Strategy {i}:")
        print(f"     ID: {strategy['id']}")
        print(f"     Metadata: {strategy['metadata']}")
        
        # Extract features to see what parameters are being used
        features = compiler.extract_features(parsed_config)
        
        print(f"\n4. Features that will be calculated:")
        for feature in features[:5]:  # Show first 5
            print(f"   - {feature.canonical_name}")
            print(f"     Params: {feature.params}")
        
        # The feature names tell us what parameters are actually being used
        # bollinger_bands_11_2.0_upper means period=11, std_dev=2.0
        # bollinger_bands_20_2.0_upper means period=20, std_dev=2.0 (default)
        
    print("\n5. Testing strategy execution with dummy data:")
    
    # Test the compiled strategy
    strategy_func = compiled_strategies[0]['function']
    
    # Create dummy features with both parameter sets
    features_11 = {
        'bollinger_bands_11_2.0_upper': 101,
        'bollinger_bands_11_2.0_lower': 99,
        'bollinger_bands_11_2.0_middle': 100
    }
    
    features_20 = {
        'bollinger_bands_20_2.0_upper': 102,
        'bollinger_bands_20_2.0_lower': 98,
        'bollinger_bands_20_2.0_middle': 100
    }
    
    # Combine both sets
    all_features = {**features_11, **features_20}
    
    bar = {'close': 100.5, 'timestamp': '2025-01-01'}
    
    # Test which features the strategy actually uses
    print("\n   Testing with price at upper band...")
    
    # Test with price above band
    bar['close'] = 101.5
    result = strategy_func(all_features, bar, {})
    
    if result:
        print(f"   Result: {result}")
        print(f"   Signal value: {result.get('signal_value')}")
    else:
        print("   No signal generated")
    
    # Now let's see which features are actually requested
    print("\n6. To definitively check which parameters are being used:")
    print("   Look at the feature names in the backtest output.")
    print("   - If you see 'bollinger_bands_11_2.0_*' → Using period=11, std_dev=2.0 ✅")
    print("   - If you see 'bollinger_bands_20_2.0_*' → Using period=20, std_dev=2.0 ❌")

if __name__ == "__main__":
    check_parameters()