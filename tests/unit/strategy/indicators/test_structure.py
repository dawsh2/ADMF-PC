"""
Unit tests for structure indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/structure.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.structure import *
from src.strategy.types import Signal

class TestStructureStrategies(unittest.TestCase):
    """Test all structure indicator strategies."""
    
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

    
    def test_pivot_points(self):
        """Test pivot_points strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(pivot_points, **{})
        
        # Test specific conditions for this strategy
        self._test_pivot_points_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(pivot_points, **{})
    
    def _test_pivot_points_conditions(self):
        """Test specific market conditions for pivot_points."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = pivot_points(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_pivot_bounces(self):
        """Test pivot_bounces strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(pivot_bounces, **{})
        
        # Test specific conditions for this strategy
        self._test_pivot_bounces_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(pivot_bounces, **{})
    
    def _test_pivot_bounces_conditions(self):
        """Test specific market conditions for pivot_bounces."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = pivot_bounces(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_fibonacci_retracement(self):
        """Test fibonacci_retracement strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(fibonacci_retracement, **{})
        
        # Test specific conditions for this strategy
        self._test_fibonacci_retracement_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(fibonacci_retracement, **{})
    
    def _test_fibonacci_retracement_conditions(self):
        """Test specific market conditions for fibonacci_retracement."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = fibonacci_retracement(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_support_resistance_breakout(self):
        """Test support_resistance_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(support_resistance_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_support_resistance_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(support_resistance_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_support_resistance_breakout_conditions(self):
        """Test specific market conditions for support_resistance_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = support_resistance_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_atr_channel_breakout(self):
        """Test atr_channel_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(atr_channel_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_atr_channel_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(atr_channel_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_atr_channel_breakout_conditions(self):
        """Test specific market conditions for atr_channel_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = atr_channel_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_price_action_swing(self):
        """Test price_action_swing strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(price_action_swing, **{})
        
        # Test specific conditions for this strategy
        self._test_price_action_swing_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(price_action_swing, **{})
    
    def _test_price_action_swing_conditions(self):
        """Test specific market conditions for price_action_swing."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = price_action_swing(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_swing_pivot_breakout(self):
        """Test swing_pivot_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(swing_pivot_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_swing_pivot_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(swing_pivot_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_swing_pivot_breakout_conditions(self):
        """Test specific market conditions for swing_pivot_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = swing_pivot_breakout(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_swing_pivot_bounce(self):
        """Test swing_pivot_bounce strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(swing_pivot_bounce, **{})
        
        # Test specific conditions for this strategy
        self._test_swing_pivot_bounce_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(swing_pivot_bounce, **{})
    
    def _test_swing_pivot_bounce_conditions(self):
        """Test specific market conditions for swing_pivot_bounce."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = swing_pivot_bounce(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_trendline_breaks(self):
        """Test trendline_breaks strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(trendline_breaks, **{})
        
        # Test specific conditions for this strategy
        self._test_trendline_breaks_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(trendline_breaks, **{})
    
    def _test_trendline_breaks_conditions(self):
        """Test specific market conditions for trendline_breaks."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = trendline_breaks(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_trendline_bounces(self):
        """Test trendline_bounces strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(trendline_bounces, **{})
        
        # Test specific conditions for this strategy
        self._test_trendline_bounces_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(trendline_bounces, **{})
    
    def _test_trendline_bounces_conditions(self):
        """Test specific market conditions for trendline_bounces."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = trendline_bounces(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_diagonal_channel_reversion(self):
        """Test diagonal_channel_reversion strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(diagonal_channel_reversion, **{'period': 14, 'oversold': 30, 'overbought': 70})
        
        # Test specific conditions for this strategy
        self._test_diagonal_channel_reversion_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(diagonal_channel_reversion, **{'period': 14, 'oversold': 30, 'overbought': 70})
    
    def _test_diagonal_channel_reversion_conditions(self):
        """Test specific market conditions for diagonal_channel_reversion."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = diagonal_channel_reversion(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_diagonal_channel_breakout(self):
        """Test diagonal_channel_breakout strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(diagonal_channel_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
        
        # Test specific conditions for this strategy
        self._test_diagonal_channel_breakout_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(diagonal_channel_breakout, **{'lookback_period': 20, 'breakout_factor': 1.01})
    
    def _test_diagonal_channel_breakout_conditions(self):
        """Test specific market conditions for diagonal_channel_breakout."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = diagonal_channel_breakout(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
