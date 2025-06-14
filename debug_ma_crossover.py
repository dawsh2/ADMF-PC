#!/usr/bin/env python3

import sys
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, './src')

from strategy.strategies.ma_crossover import ma_crossover_strategy

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_strategy_metadata():
    """Test if the strategy has proper metadata attached."""
    
    print("=== Testing MA Crossover Strategy Metadata ===")
    
    # Check if the function has metadata
    print(f"Function: {ma_crossover_strategy}")
    print(f"Has _strategy_metadata: {hasattr(ma_crossover_strategy, '_strategy_metadata')}")
    
    if hasattr(ma_crossover_strategy, '_strategy_metadata'):
        metadata = ma_crossover_strategy._strategy_metadata
        print(f"Metadata: {metadata}")
        
        feature_config = metadata.get('feature_config', {})
        print(f"Feature config: {feature_config}")
        
        # Test feature extraction logic
        params = {'fast_period': 5, 'slow_period': 100}
        print(f"Test params: {params}")
        
        required_features = []
        for feature_base, config in feature_config.items():
            param_names = config.get('params', [])
            defaults = config.get('defaults', {})
            
            print(f"Processing feature '{feature_base}': params={param_names}, defaults={defaults}")
            
            if param_names:
                # Build feature name from parameters
                feature_parts = [feature_base]
                for param_name in param_names:
                    param_value = params.get(param_name) or defaults.get(param_name) or config.get('default')
                    print(f"  {param_name}: {param_value}")
                    if param_value is not None:
                        feature_parts.append(str(param_value))
                
                if len(feature_parts) > 1:
                    feature_name = '_'.join(feature_parts)
                    required_features.append(feature_name)
                    print(f"  Built feature: {feature_name}")
                else:
                    required_features.append(feature_base)
                    print(f"  Simple feature: {feature_base}")
            else:
                # Simple feature without parameters
                required_features.append(feature_base)
                print(f"  Simple feature: {feature_base}")
        
        print(f"Required features: {required_features}")
    else:
        print("ERROR: No metadata found!")

if __name__ == "__main__":
    test_strategy_metadata()