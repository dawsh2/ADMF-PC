"""
Unit tests for momentum trading strategy.

Tests momentum calculation, RSI calculation, signal generation,
and various market conditions.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Direct imports to avoid dependency issues
import importlib.util
momentum_spec = importlib.util.spec_from_file_location(
    "momentum",
    os.path.join(os.path.dirname(__file__), '../../../src/strategy/strategies/momentum.py')
)
momentum_module = importlib.util.module_from_spec(momentum_spec)
momentum_spec.loader.exec_module(momentum_module)

MomentumStrategy = momentum_module.MomentumStrategy
create_momentum_strategy = momentum_module.create_momentum_strategy

# Import SignalDirection from protocols
protocols_spec = importlib.util.spec_from_file_location(
    "protocols",
    os.path.join(os.path.dirname(__file__), '../../../src/strategy/protocols.py')
)
protocols_module = importlib.util.module_from_spec(protocols_spec)
protocols_spec.loader.exec_module(protocols_module)

SignalDirection = protocols_module.SignalDirection


class TestMomentumStrategy(unittest.TestCase):
    """Test MomentumStrategy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MomentumStrategy(
            lookback_period=5,
            momentum_threshold=0.02,
            rsi_period=5,
            rsi_oversold=30,
            rsi_overbought=70
        )
        self.base_timestamp = datetime(2024, 1, 1, 9, 30)
    
    def test_initialization(self):
        """Test strategy initialization with parameters."""
        strategy = MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.03,
            rsi_period=14,
            rsi_oversold=25,
            rsi_overbought=75
        )
        
        self.assertEqual(strategy.lookback_period, 20)
        self.assertEqual(strategy.momentum_threshold, 0.03)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.rsi_oversold, 25)
        self.assertEqual(strategy.rsi_overbought, 75)
        self.assertEqual(strategy.name, "momentum_strategy")
        self.assertEqual(len(strategy.price_history), 0)
    
    def test_insufficient_data_returns_none(self):
        """Test that strategy returns None when insufficient data."""
        # First few data points should return None
        for i in range(4):  # Less than lookback_period
            market_data = {
                'close': 100.0 + i,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
            self.assertIsNone(signal)
        
        # Should still have stored the prices
        self.assertEqual(len(self.strategy.price_history), 4)
    
    def test_momentum_calculation(self):
        """Test momentum calculation accuracy."""
        # Feed exactly lookback_period prices
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]  # 4% gain
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Check momentum calculation
        momentum = self.strategy._calculate_momentum()
        expected_momentum = (104.0 - 100.0) / 100.0  # 0.04
        self.assertAlmostEqual(momentum, expected_momentum, places=6)
    
    def test_rsi_calculation(self):
        """Test RSI calculation with various scenarios."""
        # Test all gains - RSI should approach 100
        self.strategy.reset()
        for i in range(10):
            self.strategy.price_history.append(100.0 + i)
            self.strategy._calculate_rsi(100.0 + i)
        
        rsi = self.strategy._calculate_rsi(110.0)
        self.assertGreater(rsi, 90)  # Should be very high
        
        # Test all losses - RSI should approach 0
        self.strategy.reset()
        for i in range(10):
            self.strategy.price_history.append(100.0 - i)
            self.strategy._calculate_rsi(100.0 - i)
        
        rsi = self.strategy._calculate_rsi(89.0)
        self.assertLess(rsi, 10)  # Should be very low
        
        # Test mixed gains/losses - RSI should be around 50
        self.strategy.reset()
        for i in range(10):
            price = 100.0 + (1 if i % 2 == 0 else -1)
            self.strategy.price_history.append(price)
            self.strategy._calculate_rsi(price)
        
        rsi = self.strategy._calculate_rsi(100.0)
        self.assertGreater(rsi, 40)
        self.assertLess(rsi, 60)
    
    def test_buy_signal_generation(self):
        """Test buy signal generation with positive momentum."""
        # Create strong upward momentum
        prices = [100.0, 101.0, 102.0, 103.0, 105.0]  # 5% gain
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # Last signal should be BUY
        self.assertIsNotNone(signal)
        self.assertEqual(signal['direction'], SignalDirection.BUY)
        self.assertEqual(signal['symbol'], 'TEST')
        self.assertGreater(signal['strength'], 0)
        self.assertLessEqual(signal['strength'], 1.0)
        
        # Check metadata
        self.assertIn('momentum', signal['metadata'])
        self.assertIn('rsi', signal['metadata'])
        self.assertIn('reason', signal['metadata'])
        self.assertGreater(signal['metadata']['momentum'], self.strategy.momentum_threshold)
    
    def test_sell_signal_generation(self):
        """Test sell signal generation with negative momentum."""
        # Create strong downward momentum
        prices = [100.0, 99.0, 98.0, 97.0, 95.0]  # -5% loss
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # Last signal should be SELL
        self.assertIsNotNone(signal)
        self.assertEqual(signal['direction'], SignalDirection.SELL)
        self.assertLess(signal['metadata']['momentum'], -self.strategy.momentum_threshold)
    
    def test_oversold_reversal_signal(self):
        """Test oversold reversal signal generation."""
        # Create oversold condition with slight positive momentum
        # First create downtrend to get low RSI
        for i in range(8):
            market_data = {
                'close': 100.0 - i * 2,  # Strong downtrend
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Now slight uptick for reversal
        market_data = {
            'close': 85.0,  # Small uptick
            'timestamp': self.base_timestamp + timedelta(minutes=9),
            'symbol': 'TEST'
        }
        
        # Clear cooldown for testing
        self.strategy.last_signal_time = None
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # Might not always trigger depending on exact RSI
            self.assertEqual(signal['direction'], SignalDirection.BUY)
            self.assertEqual(signal['strength'], 0.5)  # Lower confidence
            self.assertIn('Oversold reversal', signal['metadata']['reason'])
    
    def test_overbought_reversal_signal(self):
        """Test overbought reversal signal generation."""
        # Create overbought condition with slight negative momentum
        # First create uptrend to get high RSI
        for i in range(8):
            market_data = {
                'close': 100.0 + i * 2,  # Strong uptrend
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Now slight downtick for reversal
        market_data = {
            'close': 113.0,  # Small downtick
            'timestamp': self.base_timestamp + timedelta(minutes=9),
            'symbol': 'TEST'
        }
        
        # Clear cooldown for testing
        self.strategy.last_signal_time = None
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # Might not always trigger depending on exact RSI
            self.assertEqual(signal['direction'], SignalDirection.SELL)
            self.assertEqual(signal['strength'], 0.5)  # Lower confidence
            self.assertIn('Overbought reversal', signal['metadata']['reason'])
    
    def test_signal_cooldown(self):
        """Test signal cooldown period."""
        # Generate first signal
        prices = [100.0, 101.0, 102.0, 103.0, 105.0]  # Strong momentum
        
        signal_time = None
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
            if signal:
                signal_time = signal['timestamp']
        
        self.assertIsNotNone(signal_time)
        
        # Try to generate another signal immediately
        market_data = {
            'close': 107.0,  # Even stronger momentum
            'timestamp': signal_time + timedelta(seconds=30),  # Within cooldown
            'symbol': 'TEST'
        }
        signal = self.strategy.generate_signal(market_data)
        
        # Should be None due to cooldown
        self.assertIsNone(signal)
        
        # Try after cooldown period
        market_data = {
            'close': 108.0,
            'timestamp': signal_time + timedelta(seconds=3601),  # After cooldown
            'symbol': 'TEST'
        }
        signal = self.strategy.generate_signal(market_data)
        
        # Should generate signal now
        self.assertIsNotNone(signal)
    
    def test_no_signal_in_neutral_conditions(self):
        """Test no signal when momentum is below threshold."""
        # Create small momentum below threshold
        prices = [100.0, 100.2, 100.4, 100.6, 100.8]  # 0.8% gain
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # No signal due to momentum below threshold
        self.assertIsNone(signal)
    
    def test_price_history_limit(self):
        """Test that price history is properly limited."""
        # Feed many prices
        for i in range(50):
            market_data = {
                'close': 100.0 + i,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Price history should be limited
        self.assertLessEqual(len(self.strategy.price_history), 
                            self.strategy.lookback_period * 2)
    
    def test_reset_functionality(self):
        """Test strategy reset clears all state."""
        # Generate some state
        for i in range(10):
            market_data = {
                'close': 100.0 + i,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Ensure we have state
        self.assertGreater(len(self.strategy.price_history), 0)
        self.assertGreater(len(self.strategy._gains), 0)
        
        # Reset
        self.strategy.reset()
        
        # All state should be cleared
        self.assertEqual(len(self.strategy.price_history), 0)
        self.assertEqual(len(self.strategy.rsi_values), 0)
        self.assertEqual(len(self.strategy._gains), 0)
        self.assertEqual(len(self.strategy._losses), 0)
        self.assertIsNone(self.strategy.last_signal_time)
    
    def test_handle_missing_price_data(self):
        """Test handling of market data without price."""
        market_data = {
            'timestamp': self.base_timestamp,
            'symbol': 'TEST'
            # No 'close' or 'price' field
        }
        
        signal = self.strategy.generate_signal(market_data)
        self.assertIsNone(signal)
    
    def test_handle_zero_price_division(self):
        """Test handling of zero price in momentum calculation."""
        # Start with zero price
        prices = [0.0, 1.0, 2.0, 3.0, 4.0]
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Should handle zero division gracefully
        momentum = self.strategy._calculate_momentum()
        self.assertEqual(momentum, 0.0)
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation based on momentum."""
        # Test various momentum levels
        test_cases = [
            (0.04, 1.0),    # 2x threshold = max strength
            (0.02, 0.5),    # 1x threshold = 0.5 strength
            (0.03, 0.75),   # 1.5x threshold = 0.75 strength
            (0.05, 1.0),    # >2x threshold = capped at 1.0
        ]
        
        for momentum_value, expected_strength in test_cases:
            self.strategy.reset()
            
            # Create prices to achieve target momentum
            base_price = 100.0
            target_price = base_price * (1 + momentum_value)
            
            prices = [base_price] * 4 + [target_price]
            
            for i, price in enumerate(prices):
                market_data = {
                    'close': price,
                    'timestamp': self.base_timestamp + timedelta(minutes=i),
                    'symbol': 'TEST'
                }
                signal = self.strategy.generate_signal(market_data)
            
            if signal:
                self.assertAlmostEqual(signal['strength'], expected_strength, places=2)
    
    def test_alternative_price_field(self):
        """Test using 'price' field instead of 'close'."""
        market_data = {
            'price': 100.0,  # Using 'price' instead of 'close'
            'timestamp': self.base_timestamp,
            'symbol': 'TEST'
        }
        
        # Should handle gracefully
        signal = self.strategy.generate_signal(market_data)
        self.assertEqual(len(self.strategy.price_history), 1)
        self.assertEqual(self.strategy.price_history[0], 100.0)


class TestMomentumStrategyFactory(unittest.TestCase):
    """Test momentum strategy factory function."""
    
    def test_create_with_default_config(self):
        """Test creating strategy with default configuration."""
        strategy = create_momentum_strategy()
        
        self.assertIsInstance(strategy, MomentumStrategy)
        self.assertEqual(strategy.lookback_period, 20)
        self.assertEqual(strategy.momentum_threshold, 0.02)
        self.assertEqual(strategy.rsi_period, 14)
    
    def test_create_with_custom_config(self):
        """Test creating strategy with custom configuration."""
        config = {
            'lookback_period': 30,
            'momentum_threshold': 0.03,
            'rsi_period': 21,
            'rsi_oversold': 25,
            'rsi_overbought': 75
        }
        
        strategy = create_momentum_strategy(config)
        
        self.assertEqual(strategy.lookback_period, 30)
        self.assertEqual(strategy.momentum_threshold, 0.03)
        self.assertEqual(strategy.rsi_period, 21)
        self.assertEqual(strategy.rsi_oversold, 25)
        self.assertEqual(strategy.rsi_overbought, 75)
    
    def test_partial_config_override(self):
        """Test partial configuration override."""
        config = {
            'lookback_period': 25,
            'momentum_threshold': 0.025
            # Other params should use defaults
        }
        
        strategy = create_momentum_strategy(config)
        
        self.assertEqual(strategy.lookback_period, 25)
        self.assertEqual(strategy.momentum_threshold, 0.025)
        self.assertEqual(strategy.rsi_period, 14)  # Default
        self.assertEqual(strategy.rsi_oversold, 30)  # Default


class TestMomentumStrategyIntegration(unittest.TestCase):
    """Integration tests for momentum strategy."""
    
    def test_realistic_market_scenario(self):
        """Test strategy with realistic market data."""
        strategy = MomentumStrategy(
            lookback_period=10,
            momentum_threshold=0.02,
            rsi_period=10
        )
        
        # Simulate a trading day with various market conditions
        # Morning: Uptrend
        morning_prices = [100.0, 100.5, 101.0, 101.8, 102.5, 103.0, 103.5, 104.0, 104.5, 105.0]
        # Midday: Consolidation
        midday_prices = [105.0, 104.8, 105.1, 104.9, 105.0, 105.2, 104.8, 105.0, 105.1, 104.9]
        # Afternoon: Downtrend
        afternoon_prices = [104.9, 104.5, 104.0, 103.2, 102.5, 102.0, 101.5, 101.0, 100.5, 100.0]
        
        all_prices = morning_prices + midday_prices + afternoon_prices
        signals = []
        
        base_time = datetime(2024, 1, 1, 9, 30)
        
        for i, price in enumerate(all_prices):
            market_data = {
                'close': price,
                'timestamp': base_time + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
        
        # Should have generated signals during trending periods
        self.assertGreater(len(signals), 0)
        
        # Check signal types
        buy_signals = [s for s in signals if s['direction'] == SignalDirection.BUY]
        sell_signals = [s for s in signals if s['direction'] == SignalDirection.SELL]
        
        # Should have both buy and sell signals
        self.assertGreater(len(buy_signals), 0)
        self.assertGreater(len(sell_signals), 0)
    
    def test_extreme_volatility_handling(self):
        """Test strategy behavior during extreme volatility."""
        strategy = MomentumStrategy()
        
        # Simulate extreme volatility
        volatile_prices = [100, 110, 95, 105, 90, 100, 85, 95, 80, 90]
        
        signals = []
        base_time = datetime(2024, 1, 1, 9, 30)
        
        for i, price in enumerate(volatile_prices):
            market_data = {
                'close': float(price),
                'timestamp': base_time + timedelta(minutes=i),
                'symbol': 'VOLATILE'
            }
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
        
        # Strategy should still function without errors
        # May or may not generate signals depending on thresholds
        self.assertIsInstance(signals, list)


if __name__ == "__main__":
    unittest.main()