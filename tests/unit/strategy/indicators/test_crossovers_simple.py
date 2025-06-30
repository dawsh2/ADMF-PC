#!/usr/bin/env python3
"""
Simple unit tests for crossovers indicator strategies.

Tests strategies directly without complex imports.
"""

import sys
import os
import unittest
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import the strategy decorator and feature spec directly
from src.core.components.discovery import strategy
from src.core.features.feature_spec import FeatureSpec

# Now we can import and test the actual strategy functions
# But first, let's manually parse and execute the strategy file
import importlib.util

spec = importlib.util.spec_from_file_location(
    "crossovers", 
    os.path.join(os.path.dirname(__file__), "../../../../src/strategy/strategies/indicators/crossovers.py")
)
crossovers_module = importlib.util.module_from_spec(spec)

# Set up minimal module attributes
crossovers_module.__dict__['strategy'] = strategy
crossovers_module.__dict__['FeatureSpec'] = FeatureSpec
crossovers_module.__dict__['Dict'] = Dict
crossovers_module.__dict__['Any'] = Any
crossovers_module.__dict__['Optional'] = Optional

# Execute module
try:
    spec.loader.exec_module(crossovers_module)
except Exception as e:
    print(f"Error loading module: {e}")
    # Try to load without the decorator
    with open(spec.origin, 'r') as f:
        code = f.read()
    
    # Execute the code to get the functions
    exec(code, crossovers_module.__dict__)

# Extract the strategy functions
sma_crossover = crossovers_module.sma_crossover
ema_sma_crossover = crossovers_module.ema_sma_crossover
ema_crossover = crossovers_module.ema_crossover
dema_sma_crossover = crossovers_module.dema_sma_crossover
dema_crossover = crossovers_module.dema_crossover
tema_sma_crossover = crossovers_module.tema_sma_crossover
stochastic_crossover = crossovers_module.stochastic_crossover
vortex_crossover = crossovers_module.vortex_crossover
ichimoku_cloud_position = crossovers_module.ichimoku_cloud_position


class TestCrossoversStrategies(unittest.TestCase):
    """Test crossover strategies."""
    
    def test_sma_crossover(self):
        """Test SMA crossover strategy."""
        params = {'fast_period': 10, 'slow_period': 20}
        
        # Test bullish crossover
        features = {'sma_10': 105, 'sma_20': 100}
        bar = {'close': 105, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        result = sma_crossover(features, bar, params)
        self.assertIsNotNone(result)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = {'sma_10': 95, 'sma_20': 100}
        bar = {'close': 95, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        result = sma_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test equal (no signal)
        features = {'sma_10': 100, 'sma_20': 100}
        bar = {'close': 100, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        result = sma_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        # Test missing features
        features = {}
        result = sma_crossover(features, bar, params)
        self.assertIsNone(result)
        
        print("✓ sma_crossover tests passed")
    
    def test_ema_crossover(self):
        """Test EMA crossover strategy."""
        params = {'fast_period': 10, 'slow_period': 20}
        
        # Test bullish
        features = {'ema_10': 105, 'ema_20': 100}
        bar = {'close': 105, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        result = ema_crossover(features, bar, params)
        self.assertIsNotNone(result)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish
        features = {'ema_10': 95, 'ema_20': 100}
        result = ema_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        print("✓ ema_crossover tests passed")
    
    def test_stochastic_crossover(self):
        """Test stochastic crossover strategy."""
        params = {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80}
        
        # Test bullish in oversold
        features = {'stochastic_14_3_k': 25, 'stochastic_14_3_d': 20}
        bar = {'close': 100, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        result = stochastic_crossover(features, bar, params)
        self.assertIsNotNone(result)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish in overbought
        features = {'stochastic_14_3_k': 75, 'stochastic_14_3_d': 85}
        result = stochastic_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test neutral zone (no signal)
        features = {'stochastic_14_3_k': 55, 'stochastic_14_3_d': 50}
        result = stochastic_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        print("✓ stochastic_crossover tests passed")
    
    def test_vortex_crossover(self):
        """Test vortex crossover strategy."""
        params = {'period': 14, 'threshold': 1.0}
        bar = {'close': 100, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        
        # Test bullish
        features = {'vortex_14_positive': 1.2, 'vortex_14_negative': 0.8}
        result = vortex_crossover(features, bar, params)
        self.assertIsNotNone(result)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish
        features = {'vortex_14_positive': 0.8, 'vortex_14_negative': 1.2}
        result = vortex_crossover(features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        print("✓ vortex_crossover tests passed")
    
    def test_all_strategies_return_valid_signals(self):
        """Test that all strategies return valid signal dictionaries."""
        strategies = [
            (sma_crossover, {'fast_period': 10, 'slow_period': 20}, {'sma_10': 105, 'sma_20': 100}),
            (ema_sma_crossover, {'ema_period': 10, 'sma_period': 20}, {'ema_10': 105, 'sma_20': 100}),
            (ema_crossover, {'fast_period': 10, 'slow_period': 20}, {'ema_10': 105, 'ema_20': 100}),
            (dema_sma_crossover, {'dema_period': 10, 'sma_period': 20}, {'dema_10': 105, 'sma_20': 100}),
            (dema_crossover, {'fast_period': 10, 'slow_period': 20}, {'dema_10': 105, 'dema_20': 100}),
            (tema_sma_crossover, {'tema_period': 10, 'sma_period': 20}, {'tema_10': 105, 'sma_20': 100}),
        ]
        
        bar = {'close': 100, 'symbol': 'TEST', 'timeframe': '1m', 'timestamp': '2024-01-01'}
        
        for strategy_func, params, features in strategies:
            result = strategy_func(features, bar, params)
            self.assertIsNotNone(result, f"{strategy_func.__name__} returned None")
            self.assertIn('signal_value', result)
            self.assertIn('timestamp', result)
            self.assertIn('strategy_id', result)
            self.assertIn(result['signal_value'], [-1, 0, 1])
        
        print("✓ All strategies return valid signals")


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCrossoversStrategies)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print("\n✅ All crossover strategy tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")