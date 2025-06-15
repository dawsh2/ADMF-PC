#!/usr/bin/env python3
"""Debug if strategies are being called but not generating signals."""

import logging
from src.core.components.discovery import get_component_registry
import importlib

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import strategy modules
indicator_modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.structure',
]

for module_path in indicator_modules:
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        print(f"Could not import {module_path}: {e}")

registry = get_component_registry()

# Test a specific missing strategy - bollinger_breakout
print("=== TESTING BOLLINGER_BREAKOUT STRATEGY ===\n")

strategy_info = registry.get_component('bollinger_breakout')
if strategy_info:
    print(f"Found strategy: {strategy_info.name}")
    # print(f"Module: {strategy_info.module_name}")
    print(f"Metadata: {strategy_info.metadata}")
    
    # Get the actual function
    strategy_func = strategy_info.factory
    
    # Create test data
    test_features = {
        'bollinger_bands_20_2.0_upper': 105.0,
        'bollinger_bands_20_2.0_lower': 95.0,
        'bollinger_bands_20_2.0_middle': 100.0,
        'bar_count': 100
    }
    
    test_bar = {
        'timestamp': 1234567890,
        'open': 99.0,
        'high': 101.0,
        'low': 98.0,
        'close': 100.5,
        'volume': 1000000,
        'symbol': 'SPY',
        'timeframe': '1m'
    }
    
    test_params = {
        'period': 20,
        'std_dev': 2.0
    }
    
    print(f"\nCalling strategy with test data...")
    print(f"Features: {test_features}")
    print(f"Bar: close={test_bar['close']}")
    print(f"Params: {test_params}")
    
    try:
        result = strategy_func(test_features, test_bar, test_params)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\nError calling strategy: {e}")
        import traceback
        traceback.print_exc()
else:
    print("bollinger_breakout strategy not found in registry!")

# Now check what features the strategy expects vs what it's getting
print("\n\n=== FEATURE EXPECTATIONS ===")

missing_strategies = ['bollinger_breakout', 'stochastic_crossover', 'macd_crossover', 
                     'donchian_breakout', 'ichimoku_cloud_position']

for strat_name in missing_strategies:
    strat_info = registry.get_component(strat_name)
    if strat_info:
        feature_config = strat_info.metadata.get('feature_config', [])
        print(f"\n{strat_name}:")
        print(f"  Expected features: {feature_config}")
        
        # Simulate what features would be generated
        if isinstance(feature_config, list):
            for feat_name in feature_config:
                if feat_name == 'bollinger_bands':
                    print(f"    → Would need: bollinger_bands_<period>_<std_dev>_upper/middle/lower")
                elif feat_name == 'stochastic':
                    print(f"    → Would need: stochastic_<k>_<d>_k and stochastic_<k>_<d>_d")
                elif feat_name == 'macd':
                    print(f"    → Would need: macd_<fast>_<slow>_<signal>_macd/signal/histogram")
                elif feat_name == 'donchian_channel':
                    print(f"    → Would need: donchian_channel_<period>_upper/middle/lower")
                elif feat_name == 'ichimoku':
                    print(f"    → Would need: ichimoku_<tenkan>_<kijun>_tenkan/kijun/senkou_a/senkou_b/chikou")