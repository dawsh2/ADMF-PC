"""
Unit tests for volume indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/volume.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.volume import *
from src.strategy.types import Signal

class TestVolumeStrategies(unittest.TestCase):
    """Test all volume indicator strategies."""
    
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

    
    def test_obv_trend(self):
        """Test obv_trend strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(obv_trend, **{})
        
        # Test specific conditions for this strategy
        self._test_obv_trend_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(obv_trend, **{})
    
    def _test_obv_trend_conditions(self):
        """Test specific market conditions for obv_trend."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = obv_trend(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_mfi_bands(self):
        """Test mfi_bands strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(mfi_bands, **{})
        
        # Test specific conditions for this strategy
        self._test_mfi_bands_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(mfi_bands, **{})
    
    def _test_mfi_bands_conditions(self):
        """Test specific market conditions for mfi_bands."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = mfi_bands(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_vwap_deviation(self):
        """Test vwap_deviation strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(vwap_deviation, **{})
        
        # Test specific conditions for this strategy
        self._test_vwap_deviation_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(vwap_deviation, **{})
    
    def _test_vwap_deviation_conditions(self):
        """Test specific market conditions for vwap_deviation."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = vwap_deviation(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_chaikin_money_flow(self):
        """Test chaikin_money_flow strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(chaikin_money_flow, **{})
        
        # Test specific conditions for this strategy
        self._test_chaikin_money_flow_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(chaikin_money_flow, **{})
    
    def _test_chaikin_money_flow_conditions(self):
        """Test specific market conditions for chaikin_money_flow."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = chaikin_money_flow(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_accumulation_distribution(self):
        """Test accumulation_distribution strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(accumulation_distribution, **{})
        
        # Test specific conditions for this strategy
        self._test_accumulation_distribution_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(accumulation_distribution, **{})
    
    def _test_accumulation_distribution_conditions(self):
        """Test specific market conditions for accumulation_distribution."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = accumulation_distribution(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
