"""
Unit tests for trend indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/trend.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.trend import *
from src.strategy.types import Signal

class TestTrendStrategies(unittest.TestCase):
    """Test all trend indicator strategies."""
    
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

    
    def test_adx_trend_strength(self):
        """Test adx_trend_strength strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(adx_trend_strength, **{})
        
        # Test specific conditions for this strategy
        self._test_adx_trend_strength_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(adx_trend_strength, **{})
    
    def _test_adx_trend_strength_conditions(self):
        """Test specific market conditions for adx_trend_strength."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = adx_trend_strength(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_parabolic_sar(self):
        """Test parabolic_sar strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(parabolic_sar, **{})
        
        # Test specific conditions for this strategy
        self._test_parabolic_sar_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(parabolic_sar, **{})
    
    def _test_parabolic_sar_conditions(self):
        """Test specific market conditions for parabolic_sar."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = parabolic_sar(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_aroon_crossover(self):
        """Test aroon_crossover strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(aroon_crossover, **{})
        
        # Test specific conditions for this strategy
        self._test_aroon_crossover_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(aroon_crossover, **{})
    
    def _test_aroon_crossover_conditions(self):
        """Test specific market conditions for aroon_crossover."""
        # Create specific market conditions based on strategy type
        # Test bullish crossover
        bullish_data = self.data.copy()
        bullish_data['close'] = np.concatenate([
            np.linspace(100, 90, 50),  # Downtrend
            np.linspace(90, 110, 50)   # Uptrend
        ])
        signal = aroon_crossover(bullish_data)
        # Should generate buy signal during crossover

    
    def test_supertrend(self):
        """Test supertrend strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(supertrend, **{})
        
        # Test specific conditions for this strategy
        self._test_supertrend_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(supertrend, **{})
    
    def _test_supertrend_conditions(self):
        """Test specific market conditions for supertrend."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = supertrend(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_linear_regression_slope(self):
        """Test linear_regression_slope strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(linear_regression_slope, **{})
        
        # Test specific conditions for this strategy
        self._test_linear_regression_slope_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(linear_regression_slope, **{})
    
    def _test_linear_regression_slope_conditions(self):
        """Test specific market conditions for linear_regression_slope."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = linear_regression_slope(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
