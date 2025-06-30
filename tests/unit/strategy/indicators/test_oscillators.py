"""
Unit tests for oscillators indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/oscillators.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.oscillators import *
from src.strategy.types import Signal

class TestOscillatorsStrategies(unittest.TestCase):
    """Test all oscillators indicator strategies."""
    
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

    
    def test_rsi_threshold(self):
        """Test rsi_threshold strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(rsi_threshold, **{'period': 14, 'oversold': 30, 'overbought': 70})
        
        # Test specific conditions for this strategy
        self._test_rsi_threshold_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(rsi_threshold, **{'period': 14, 'oversold': 30, 'overbought': 70})
    
    def _test_rsi_threshold_conditions(self):
        """Test specific market conditions for rsi_threshold."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = rsi_threshold(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_rsi_bands(self):
        """Test rsi_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(rsi_bands, **{'period': 14, 'oversold': 30, 'overbought': 70})
        
        # Test specific conditions for this strategy
        self._test_rsi_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(rsi_bands, **{'period': 14, 'oversold': 30, 'overbought': 70})
    
    def _test_rsi_bands_conditions(self):
        """Test specific market conditions for rsi_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = rsi_bands(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_cci_threshold(self):
        """Test cci_threshold strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(cci_threshold, **{})
        
        # Test specific conditions for this strategy
        self._test_cci_threshold_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(cci_threshold, **{})
    
    def _test_cci_threshold_conditions(self):
        """Test specific market conditions for cci_threshold."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = cci_threshold(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_cci_bands(self):
        """Test cci_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(cci_bands, **{})
        
        # Test specific conditions for this strategy
        self._test_cci_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(cci_bands, **{})
    
    def _test_cci_bands_conditions(self):
        """Test specific market conditions for cci_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = cci_bands(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_stochastic_rsi(self):
        """Test stochastic_rsi strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(stochastic_rsi, **{'period': 14, 'oversold': 30, 'overbought': 70})
        
        # Test specific conditions for this strategy
        self._test_stochastic_rsi_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(stochastic_rsi, **{'period': 14, 'oversold': 30, 'overbought': 70})
    
    def _test_stochastic_rsi_conditions(self):
        """Test specific market conditions for stochastic_rsi."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = stochastic_rsi(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_williams_r(self):
        """Test williams_r strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(williams_r, **{})
        
        # Test specific conditions for this strategy
        self._test_williams_r_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(williams_r, **{})
    
    def _test_williams_r_conditions(self):
        """Test specific market conditions for williams_r."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = williams_r(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_roc_threshold(self):
        """Test roc_threshold strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(roc_threshold, **{})
        
        # Test specific conditions for this strategy
        self._test_roc_threshold_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(roc_threshold, **{})
    
    def _test_roc_threshold_conditions(self):
        """Test specific market conditions for roc_threshold."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = roc_threshold(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_ultimate_oscillator(self):
        """Test ultimate_oscillator strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(ultimate_oscillator, **{'fast_period': 10, 'slow_period': 20})
        
        # Test specific conditions for this strategy
        self._test_ultimate_oscillator_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(ultimate_oscillator, **{'fast_period': 10, 'slow_period': 20})
    
    def _test_ultimate_oscillator_conditions(self):
        """Test specific market conditions for ultimate_oscillator."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = ultimate_oscillator(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
