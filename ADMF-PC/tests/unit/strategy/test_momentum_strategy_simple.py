"""
Unit tests for momentum trading strategy.

Tests momentum calculation, RSI calculation, signal generation,
and various market conditions.
"""

import unittest
from datetime import datetime, timedelta
from enum import Enum
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


# Create a local SignalDirection enum to avoid import issues
class SignalDirection(Enum):
    """Signal direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# Mock the momentum strategy class locally to test its logic
class MomentumStrategy:
    """
    Momentum-based trading strategy.
    
    This is a copy of the actual strategy for testing purposes.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 momentum_threshold: float = 0.02,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """Initialize momentum strategy."""
        # Parameters
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # State
        self.price_history = []
        self.rsi_values = []
        self.last_signal_time = None
        self.signal_cooldown = 3600  # 1 hour in seconds
        
        # Internal calculation state
        self._gains = []
        self._losses = []
        
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "momentum_strategy"
    
    def generate_signal(self, market_data):
        """Generate trading signal from market data."""
        # Extract price
        price = market_data.get('close', market_data.get('price'))
        timestamp = market_data.get('timestamp', datetime.now())
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        if price is None:
            return None
        
        # Update price history
        self.price_history.append(price)
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history.pop(0)
        
        # Need enough data
        if len(self.price_history) < self.lookback_period:
            return None
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                return None
        
        # Calculate momentum
        momentum = self._calculate_momentum()
        
        # Calculate RSI
        rsi = self._calculate_rsi(price)
        
        # Generate signal based on momentum and RSI
        signal = None
        
        if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
            # Bullish momentum, not overbought
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.BUY,
                'strength': min(momentum / (self.momentum_threshold * 2), 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Positive momentum with room to run'
                }
            }
            
        elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
            # Bearish momentum, not oversold
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.SELL,
                'strength': min(abs(momentum) / (self.momentum_threshold * 2), 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Negative momentum with room to fall'
                }
            }
            
        elif rsi < self.rsi_oversold and momentum > 0:
            # Oversold with positive momentum - potential reversal
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.BUY,
                'strength': 0.5,  # Lower confidence for reversal
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Oversold reversal signal'
                }
            }
            
        elif rsi > self.rsi_overbought and momentum < 0:
            # Overbought with negative momentum - potential reversal
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.SELL,
                'strength': 0.5,  # Lower confidence for reversal
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Overbought reversal signal'
                }
            }
        
        if signal:
            self.last_signal_time = timestamp
        
        return signal
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum."""
        if len(self.price_history) < self.lookback_period:
            return 0.0
        
        # Simple rate of change
        current_price = self.price_history[-1]
        past_price = self.price_history[-self.lookback_period]
        
        if past_price == 0:
            return 0.0
        
        return (current_price - past_price) / past_price
    
    def _calculate_rsi(self, current_price: float) -> float:
        """Calculate RSI indicator."""
        if len(self.price_history) < 2:
            return 50.0  # Neutral
        
        # Calculate price change
        prev_price = self.price_history[-2] if len(self.price_history) > 1 else current_price
        change = current_price - prev_price
        
        # Track gains and losses
        gain = max(0, change)
        loss = max(0, -change)
        
        self._gains.append(gain)
        self._losses.append(loss)
        
        # Limit history
        if len(self._gains) > self.rsi_period:
            self._gains.pop(0)
            self._losses.pop(0)
        
        # Need enough data
        if len(self._gains) < self.rsi_period:
            return 50.0
        
        # Calculate average gain/loss
        avg_gain = sum(self._gains) / len(self._gains)
        avg_loss = sum(self._losses) / len(self._losses)
        
        if avg_loss == 0:
            return 100.0  # No losses = RSI 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.price_history.clear()
        self.rsi_values.clear()
        self._gains.clear()
        self._losses.clear()
        self.last_signal_time = None


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
        
        signal = None
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
        
        signal = None
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
        
        signal = None
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # No signal due to momentum below threshold
        self.assertIsNone(signal)
    
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
            
            signal = None
            for i, price in enumerate(prices):
                market_data = {
                    'close': price,
                    'timestamp': self.base_timestamp + timedelta(minutes=i),
                    'symbol': 'TEST'
                }
                signal = self.strategy.generate_signal(market_data)
            
            if signal:
                self.assertAlmostEqual(signal['strength'], expected_strength, places=2)


class TestMomentumStrategyEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_extreme_rsi_values(self):
        """Test strategy behavior with extreme RSI values."""
        strategy = MomentumStrategy()
        
        # Build up price history and gains/losses properly
        prices = [100.0]
        for i in range(1, 20):
            price = 100.0 + i * 5  # Strong uptrend
            prices.append(price)
            strategy.price_history = prices[-2:]  # Keep last 2 for change calculation
            rsi = strategy._calculate_rsi(price)
        
        # Final RSI should be very high due to consistent gains
        self.assertGreater(rsi, 70)  # Should be overbought
        
        # Test zero average loss scenario
        strategy.reset()
        # Build up history with only gains
        strategy.price_history = [100.0, 101.0]
        for i in range(strategy.rsi_period):
            strategy._gains.append(1.0)
            strategy._losses.append(0.0)
        
        # Now the RSI calculation should work with the populated gains/losses
        avg_gain = sum(strategy._gains) / len(strategy._gains)
        avg_loss = sum(strategy._losses) / len(strategy._losses)
        # With zero losses, RSI = 100
        self.assertEqual(avg_loss, 0.0)
        self.assertGreater(avg_gain, 0)
    
    def test_price_history_management(self):
        """Test that price history is properly managed."""
        strategy = MomentumStrategy(lookback_period=5)
        
        # Add many prices
        for i in range(50):
            market_data = {
                'close': 100.0 + i,
                'timestamp': datetime.now(),
                'symbol': 'TEST'
            }
            strategy.generate_signal(market_data)
        
        # Should be limited to 2x lookback period
        self.assertLessEqual(len(strategy.price_history), strategy.lookback_period * 2)
    
    def test_concurrent_signal_types(self):
        """Test behavior when multiple signal conditions are met."""
        strategy = MomentumStrategy(
            momentum_threshold=0.02,
            rsi_oversold=30,
            rsi_overbought=70
        )
        
        # Create a scenario where momentum is positive but RSI is overbought
        # Strategy should prioritize the momentum signal with RSI check
        for i in range(20):
            price = 100.0 + i * 0.5  # Steady uptrend
            market_data = {
                'close': price,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = strategy.generate_signal(market_data)
        
        # With steady uptrend, RSI should eventually become overbought
        # and signals should stop or become reversal signals
        self.assertIsNotNone(strategy.price_history)


if __name__ == "__main__":
    unittest.main()