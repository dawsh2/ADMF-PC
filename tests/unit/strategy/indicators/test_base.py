"""
Base test utilities for indicator strategies.

Provides mock data and test helpers that don't require pandas.
"""

import unittest
from typing import Dict, Any, List
from datetime import datetime, timedelta


class IndicatorStrategyTestBase(unittest.TestCase):
    """Base class for indicator strategy tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_timestamp = datetime(2024, 1, 1, 9, 30)
        
    def create_bar(self, index: int, price: float, volume: int = 1000000) -> Dict[str, Any]:
        """Create a single bar of market data."""
        return {
            'timestamp': self.base_timestamp + timedelta(minutes=index),
            'open': price * 0.99,
            'high': price * 1.01,
            'low': price * 0.98,
            'close': price,
            'volume': volume,
            'symbol': 'TEST',
            'timeframe': '1m'
        }
    
    def create_features(self, **kwargs) -> Dict[str, Any]:
        """Create a features dictionary with indicator values."""
        return kwargs
    
    def create_trending_prices(self, start: float = 100, trend: float = 0.001, length: int = 100) -> List[float]:
        """Create a list of trending prices."""
        return [start * (1 + trend) ** i for i in range(length)]
    
    def create_ranging_prices(self, center: float = 100, amplitude: float = 2, length: int = 100) -> List[float]:
        """Create a list of ranging prices."""
        import math
        return [center + amplitude * math.sin(2 * math.pi * i / 20) for i in range(length)]
    
    def test_strategy_returns_dict(self, strategy_func, features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]):
        """Test that strategy returns a properly formatted dictionary."""
        result = strategy_func(features, bar, params)
        
        # Should return a dictionary
        self.assertIsInstance(result, dict)
        
        # Should have required fields
        self.assertIn('signal_value', result)
        self.assertIn('timestamp', result)
        self.assertIn('strategy_id', result)
        self.assertIn('symbol_timeframe', result)
        self.assertIn('metadata', result)
        
        # Signal value should be -1, 0, or 1
        self.assertIn(result['signal_value'], [-1, 0, 1])
        
        return result
    
    def test_strategy_with_missing_features(self, strategy_func, params: Dict[str, Any]):
        """Test strategy behavior with missing features."""
        features = {}  # Empty features
        bar = self.create_bar(0, 100)
        
        result = strategy_func(features, bar, params)
        
        # Should return None or signal_value of 0
        if result is not None:
            self.assertEqual(result['signal_value'], 0)
    
    def test_strategy_with_none_values(self, strategy_func, feature_keys: List[str], params: Dict[str, Any]):
        """Test strategy behavior when features have None values."""
        features = {key: None for key in feature_keys}
        bar = self.create_bar(0, 100)
        
        result = strategy_func(features, bar, params)
        
        # Should handle None gracefully
        if result is not None:
            self.assertEqual(result['signal_value'], 0)