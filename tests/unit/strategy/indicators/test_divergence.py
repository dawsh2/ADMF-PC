"""
Unit tests for divergence indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/divergence.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.divergence import *
from src.strategy.types import Signal

class TestDivergenceStrategies(unittest.TestCase):
    """Test all divergence indicator strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with various market conditions
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Base price with trend
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        
        self.prices = base_price + trend + noise
        
        # Create DataFrame with OHLCV data
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'close': self.prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        # Add some indicators that strategies might need
        self.data['returns'] = self.data['close'].pct_change()
        
    def _test_strategy_basic(self, strategy_func, **kwargs):
        """Basic test template for any strategy function."""
        # Test with valid data
        signal = strategy_func(self.data, **kwargs)
        
        # Check signal is valid
        self.assertIsInstance(signal, Signal)
        self.assertIn(signal.direction, [-1, 0, 1])
        self.assertIsInstance(signal.magnitude, (int, float))
        self.assertGreaterEqual(signal.magnitude, 0)
        self.assertLessEqual(signal.magnitude, 1)
        
        # Test with empty data
        empty_df = pd.DataFrame()
        signal = strategy_func(empty_df, **kwargs)
        self.assertEqual(signal.direction, 0)
        
        # Test with insufficient data
        small_df = self.data.head(2)
        signal = strategy_func(small_df, **kwargs)
        self.assertEqual(signal.direction, 0)
        
        return signal
    
    def _test_strategy_edge_cases(self, strategy_func, **kwargs):
        """Test edge cases for a strategy."""
        # Test with NaN values
        data_with_nan = self.data.copy()
        data_with_nan.loc[50:55, 'close'] = np.nan
        signal = strategy_func(data_with_nan, **kwargs)
        self.assertIsInstance(signal, Signal)
        
        # Test with extreme values
        extreme_data = self.data.copy()
        extreme_data['close'] = extreme_data['close'] * 1000
        signal = strategy_func(extreme_data, **kwargs)
        self.assertIsInstance(signal, Signal)
        
        # Test with zero volume
        zero_vol_data = self.data.copy()
        zero_vol_data['volume'] = 0
        signal = strategy_func(zero_vol_data, **kwargs)
        self.assertIsInstance(signal, Signal)

    
    def test_rsi_divergence(self):
        """Test rsi_divergence strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(rsi_divergence, **{'period': 14, 'oversold': 30, 'overbought': 70})
        
        # Test specific conditions for this strategy
        self._test_rsi_divergence_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(rsi_divergence, **{'period': 14, 'oversold': 30, 'overbought': 70})
    
    def _test_rsi_divergence_conditions(self):
        """Test specific market conditions for rsi_divergence."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = rsi_divergence(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_macd_histogram_divergence(self):
        """Test macd_histogram_divergence strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(macd_histogram_divergence, **{'fast_period': 10, 'slow_period': 20})
        
        # Test specific conditions for this strategy
        self._test_macd_histogram_divergence_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(macd_histogram_divergence, **{'fast_period': 10, 'slow_period': 20})
    
    def _test_macd_histogram_divergence_conditions(self):
        """Test specific market conditions for macd_histogram_divergence."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = macd_histogram_divergence(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_stochastic_divergence(self):
        """Test stochastic_divergence strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(stochastic_divergence, **{})
        
        # Test specific conditions for this strategy
        self._test_stochastic_divergence_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(stochastic_divergence, **{})
    
    def _test_stochastic_divergence_conditions(self):
        """Test specific market conditions for stochastic_divergence."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = stochastic_divergence(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_momentum_divergence(self):
        """Test momentum_divergence strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(momentum_divergence, **{'lookback_period': 20, 'threshold': 0.02})
        
        # Test specific conditions for this strategy
        self._test_momentum_divergence_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(momentum_divergence, **{'lookback_period': 20, 'threshold': 0.02})
    
    def _test_momentum_divergence_conditions(self):
        """Test specific market conditions for momentum_divergence."""
        # Create specific market conditions based on strategy type
        # Test strong momentum
        momentum_data = self.data.copy()
        momentum_data['close'] = 100 * (1.1 ** np.arange(100))  # Strong uptrend
        signal = momentum_divergence(momentum_data)
        self.assertEqual(signal.direction, 1)  # Should be bullish

    
    def test_obv_price_divergence(self):
        """Test obv_price_divergence strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(obv_price_divergence, **{})
        
        # Test specific conditions for this strategy
        self._test_obv_price_divergence_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(obv_price_divergence, **{})
    
    def _test_obv_price_divergence_conditions(self):
        """Test specific market conditions for obv_price_divergence."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = obv_price_divergence(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
