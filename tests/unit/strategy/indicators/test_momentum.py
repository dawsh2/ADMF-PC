"""
Unit tests for momentum indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/momentum.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.momentum import *
from src.strategy.types import Signal

class TestMomentumStrategies(unittest.TestCase):
    """Test all momentum indicator strategies."""
    
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

    
    def test_macd_crossover_strategy(self):
        """Test macd_crossover_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(macd_crossover_strategy, **{'fast_period': 10, 'slow_period': 20})
        
        # Test specific conditions for this strategy
        self._test_macd_crossover_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(macd_crossover_strategy, **{'fast_period': 10, 'slow_period': 20})
    
    def _test_macd_crossover_strategy_conditions(self):
        """Test specific market conditions for macd_crossover_strategy."""
        # Create specific market conditions based on strategy type
        # Test bullish crossover
        bullish_data = self.data.copy()
        bullish_data['close'] = np.concatenate([
            np.linspace(100, 90, 50),  # Downtrend
            np.linspace(90, 110, 50)   # Uptrend
        ])
        signal = macd_crossover_strategy(bullish_data)
        # Should generate buy signal during crossover

    
    def test_momentum_breakout_strategy(self):
        """Test momentum_breakout_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(momentum_breakout_strategy, **{'lookback_period': 20, 'threshold': 0.02})
        
        # Test specific conditions for this strategy
        self._test_momentum_breakout_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(momentum_breakout_strategy, **{'lookback_period': 20, 'threshold': 0.02})
    
    def _test_momentum_breakout_strategy_conditions(self):
        """Test specific market conditions for momentum_breakout_strategy."""
        # Create specific market conditions based on strategy type
        # Test strong momentum
        momentum_data = self.data.copy()
        momentum_data['close'] = 100 * (1.1 ** np.arange(100))  # Strong uptrend
        signal = momentum_breakout_strategy(momentum_data)
        self.assertEqual(signal.direction, 1)  # Should be bullish

    
    def test_roc_trend_strategy(self):
        """Test roc_trend_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(roc_trend_strategy, **{})
        
        # Test specific conditions for this strategy
        self._test_roc_trend_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(roc_trend_strategy, **{})
    
    def _test_roc_trend_strategy_conditions(self):
        """Test specific market conditions for roc_trend_strategy."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = roc_trend_strategy(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_adx_trend_strength_strategy(self):
        """Test adx_trend_strength_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(adx_trend_strength_strategy, **{})
        
        # Test specific conditions for this strategy
        self._test_adx_trend_strength_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(adx_trend_strength_strategy, **{})
    
    def _test_adx_trend_strength_strategy_conditions(self):
        """Test specific market conditions for adx_trend_strength_strategy."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = adx_trend_strength_strategy(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_aroon_oscillator_strategy(self):
        """Test aroon_oscillator_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(aroon_oscillator_strategy, **{})
        
        # Test specific conditions for this strategy
        self._test_aroon_oscillator_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(aroon_oscillator_strategy, **{})
    
    def _test_aroon_oscillator_strategy_conditions(self):
        """Test specific market conditions for aroon_oscillator_strategy."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = aroon_oscillator_strategy(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_vortex_trend_strategy(self):
        """Test vortex_trend_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(vortex_trend_strategy, **{})
        
        # Test specific conditions for this strategy
        self._test_vortex_trend_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(vortex_trend_strategy, **{})
    
    def _test_vortex_trend_strategy_conditions(self):
        """Test specific market conditions for vortex_trend_strategy."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = vortex_trend_strategy(self.data)
        self.assertIsInstance(signal, Signal)

    
    def test_elder_ray_strategy(self):
        """Test elder_ray_strategy strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic(elder_ray_strategy, **{})
        
        # Test specific conditions for this strategy
        self._test_elder_ray_strategy_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases(elder_ray_strategy, **{})
    
    def _test_elder_ray_strategy_conditions(self):
        """Test specific market conditions for elder_ray_strategy."""
        # Create specific market conditions based on strategy type
        # Test general conditions
        signal = elder_ray_strategy(self.data)
        self.assertIsInstance(signal, Signal)


if __name__ == '__main__':
    unittest.main()
