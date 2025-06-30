"""
Unit tests for crossovers indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/crossovers.py
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Mock pandas and numpy before importing strategies
import tests.unit.strategy.indicators.mock_dependencies

from tests.unit.strategy.indicators.test_base import IndicatorStrategyTestBase
from src.strategy.strategies.indicators.crossovers import (
    sma_crossover, ema_sma_crossover, ema_crossover,
    dema_sma_crossover, dema_crossover, tema_sma_crossover,
    stochastic_crossover, vortex_crossover, ichimoku_cloud_position
)


class TestCrossoversStrategies(IndicatorStrategyTestBase):
    """Test all crossovers indicator strategies."""
    
    def test_sma_crossover(self):
        """Test sma_crossover strategy."""
        params = {'fast_period': 10, 'slow_period': 20}
        
        # Test bullish crossover (fast > slow)
        features = self.create_features(sma_10=105, sma_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover (fast < slow)
        features = self.create_features(sma_10=95, sma_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test equal (no signal)
        features = self.create_features(sma_10=100, sma_20=100)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        # Test missing features
        self.test_strategy_with_missing_features(sma_crossover, params)
        self.test_strategy_with_none_values(sma_crossover, ['sma_10', 'sma_20'], params)
    
    def test_ema_sma_crossover(self):
        """Test ema_sma_crossover strategy."""
        params = {'ema_period': 10, 'sma_period': 20}
        
        # Test bullish crossover
        features = self.create_features(ema_10=105, sma_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(ema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(ema_10=95, sma_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(ema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test missing features
        self.test_strategy_with_missing_features(ema_sma_crossover, params)
    
    def test_ema_crossover(self):
        """Test ema_crossover strategy."""
        params = {'fast_period': 10, 'slow_period': 20}
        
        # Test bullish crossover
        features = self.create_features(ema_10=105, ema_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(ema_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(ema_10=95, ema_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(ema_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test missing features
        self.test_strategy_with_missing_features(ema_crossover, params)
    
    def test_dema_sma_crossover(self):
        """Test dema_sma_crossover strategy."""
        params = {'dema_period': 10, 'sma_period': 20}
        
        # Test bullish crossover
        features = self.create_features(dema_10=105, sma_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(dema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(dema_10=95, sma_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(dema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test missing features
        self.test_strategy_with_missing_features(dema_sma_crossover, params)
    
    def test_dema_crossover(self):
        """Test dema_crossover strategy."""
        params = {'fast_period': 10, 'slow_period': 20}
        
        # Test bullish crossover
        features = self.create_features(dema_10=105, dema_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(dema_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(dema_10=95, dema_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(dema_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test missing features
        self.test_strategy_with_missing_features(dema_crossover, params)
    
    def test_tema_sma_crossover(self):
        """Test tema_sma_crossover strategy."""
        params = {'tema_period': 10, 'sma_period': 20}
        
        # Test bullish crossover
        features = self.create_features(tema_10=105, sma_20=100)
        bar = self.create_bar(0, 105)
        result = self.test_strategy_returns_dict(tema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(tema_10=95, sma_20=100)
        bar = self.create_bar(0, 95)
        result = self.test_strategy_returns_dict(tema_sma_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test missing features
        self.test_strategy_with_missing_features(tema_sma_crossover, params)
    
    def test_stochastic_crossover(self):
        """Test stochastic_crossover strategy."""
        params = {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80}
        
        # Test bullish crossover in oversold zone
        features = self.create_features(stochastic_14_3_k=25, stochastic_14_3_d=20)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(stochastic_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover in overbought zone
        features = self.create_features(stochastic_14_3_k=75, stochastic_14_3_d=85)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(stochastic_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test crossover in neutral zone (no signal)
        features = self.create_features(stochastic_14_3_k=55, stochastic_14_3_d=50)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(stochastic_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        # Test missing features
        self.test_strategy_with_missing_features(stochastic_crossover, params)
    
    def test_vortex_crossover(self):
        """Test vortex_crossover strategy."""
        params = {'period': 14, 'threshold': 1.0}
        
        # Test bullish crossover
        features = self.create_features(vortex_14_positive=1.2, vortex_14_negative=0.8)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(vortex_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test bearish crossover
        features = self.create_features(vortex_14_positive=0.8, vortex_14_negative=1.2)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(vortex_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test weak signal (no signal)
        features = self.create_features(vortex_14_positive=0.95, vortex_14_negative=0.9)
        bar = self.create_bar(0, 100)
        result = self.test_strategy_returns_dict(vortex_crossover, features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        # Test missing features
        self.test_strategy_with_missing_features(vortex_crossover, params)
    
    def test_ichimoku_cloud_position(self):
        """Test ichimoku_cloud_position strategy."""
        params = {'tenkan_period': 9, 'kijun_period': 26}
        
        # Test price above cloud (bullish)
        features = self.create_features(
            ichimoku_9_26_tenkan=105,
            ichimoku_9_26_kijun=100,
            ichimoku_9_26_senkou_a=102,
            ichimoku_9_26_senkou_b=98
        )
        bar = self.create_bar(0, 110)  # Price above cloud
        result = self.test_strategy_returns_dict(ichimoku_cloud_position, features, bar, params)
        self.assertEqual(result['signal_value'], 1)
        
        # Test price below cloud (bearish)
        features = self.create_features(
            ichimoku_9_26_tenkan=95,
            ichimoku_9_26_kijun=100,
            ichimoku_9_26_senkou_a=102,
            ichimoku_9_26_senkou_b=98
        )
        bar = self.create_bar(0, 90)  # Price below cloud
        result = self.test_strategy_returns_dict(ichimoku_cloud_position, features, bar, params)
        self.assertEqual(result['signal_value'], -1)
        
        # Test price inside cloud (neutral)
        features = self.create_features(
            ichimoku_9_26_tenkan=100,
            ichimoku_9_26_kijun=100,
            ichimoku_9_26_senkou_a=102,
            ichimoku_9_26_senkou_b=98
        )
        bar = self.create_bar(0, 100)  # Price inside cloud
        result = self.test_strategy_returns_dict(ichimoku_cloud_position, features, bar, params)
        self.assertEqual(result['signal_value'], 0)
        
        # Test missing features
        self.test_strategy_with_missing_features(ichimoku_cloud_position, params)


if __name__ == '__main__':
    unittest.main()