"""
Unit tests for volatility indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/volatility.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.volatility import *
from src.strategy.types import Signal

class TestVolatilityStrategies(unittest.TestCase):
    """Test all volatility indicator strategies."""
    
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

    
    def test_keltner_breakout(self):
        """Test keltner_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(keltner_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_keltner_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(keltner_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_keltner_breakout_conditions(self):
        """Test specific market conditions for keltner_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = keltner_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_donchian_breakout(self):
        """Test donchian_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(donchian_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_donchian_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(donchian_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_donchian_breakout_conditions(self):
        """Test specific market conditions for donchian_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = donchian_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_bollinger_bands(self):
        """Test bollinger_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(bollinger_bands, **{'period': 20, 'num_std': 2})
        
        # Test specific conditions for this strategy
        self._test_bollinger_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(bollinger_bands, **{'period': 20, 'num_std': 2})
    
    def _test_bollinger_bands_conditions(self):
        """Test specific market conditions for bollinger_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = bollinger_bands(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_bollinger_breakout(self):
        """Test bollinger_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(bollinger_breakout, **{'period': 20, 'num_std': 2})
        
        # Test specific conditions for this strategy
        self._test_bollinger_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(bollinger_breakout, **{'period': 20, 'num_std': 2})
    
    def _test_bollinger_breakout_conditions(self):
        """Test specific market conditions for bollinger_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = bollinger_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_keltner_bands(self):
        """Test keltner_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(keltner_bands, **{})
        
        # Test specific conditions for this strategy
        self._test_keltner_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(keltner_bands, **{})
    
    def _test_keltner_bands_conditions(self):
        """Test specific market conditions for keltner_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = keltner_bands(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_donchian_bands(self):
        """Test donchian_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(donchian_bands, **{})
        
        # Test specific conditions for this strategy
        self._test_donchian_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(donchian_bands, **{})
    
    def _test_donchian_bands_conditions(self):
        """Test specific market conditions for donchian_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = donchian_bands(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
