#!/usr/bin/env python3
"""
Simple test to verify parameters are being passed correctly to strategies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.compiler import StrategyCompiler

def test_parameter_passing():
    """Test that parameters are correctly passed to strategies."""
    compiler = StrategyCompiler()
    
    print("Testing Parameter Fix for Bollinger Bands")
    print("=" * 50)
    
    # Your current config
    config = {
        'strategy': [
            {'bollinger_bands': {'period': 11, 'std_dev': 2.0}}
        ]
    }
    
    print("\nCompiling strategy with period=11, std_dev=2.0...")
    
    try:
        compiled = compiler.compile_strategies(config)
        strategy_func = compiled[0]['function']
        
        # Test execution with dummy data
        features = {
            'bollinger_bands_11_2.0_upper': 101,
            'bollinger_bands_11_2.0_lower': 99,
            'bollinger_bands_11_2.0_middle': 100
        }
        bar = {'close': 100}
        
        print("\nTesting strategy execution...")
        result = strategy_func(features, bar, {})
        
        print("✅ Strategy executed successfully!")
        print(f"   Strategy metadata: {compiled[0]['metadata']}")
        
        # Extract features to see what the strategy expects
        print("\nFeatures the strategy expects:")
        features_list = compiler.extract_features(config)
        for feature in features_list:
            print(f"   - {feature.canonical_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_passing()