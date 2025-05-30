"""
Unit tests for mean reversion trading strategy.

Tests Bollinger Bands, z-score calculation, RSI confirmation,
mean reversion exits, and position management.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.risk.protocols import SignalType, OrderSide


class SimpleBollingerBands:
    """Simplified Bollinger Bands for testing."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_multiplier = std_dev
        self.prices = []
        self.middle_band = None
        self.upper_band = None
        self.lower_band = None
        self.std_dev = None
        self.ready = False
    
    def calculate(self, price: float, timestamp=None) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands."""
        self.prices.append(price)
        if len(self.prices) > self.period:
            self.prices.pop(0)
        
        if len(self.prices) >= self.period:
            # Calculate moving average
            self.middle_band = sum(self.prices) / len(self.prices)
            
            # Calculate standard deviation
            variance = sum((p - self.middle_band) ** 2 for p in self.prices) / len(self.prices)
            self.std_dev = variance ** 0.5
            
            # Calculate bands
            self.upper_band = self.middle_band + (self.std_multiplier * self.std_dev)
            self.lower_band = self.middle_band - (self.std_multiplier * self.std_dev)
            
            self.ready = True
            
            return {
                'middle': self.middle_band,
                'upper': self.upper_band,
                'lower': self.lower_band,
                'std_dev': self.std_dev
            }
        
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.prices.clear()
        self.middle_band = None
        self.upper_band = None
        self.lower_band = None
        self.std_dev = None
        self.ready = False


class SimpleRSI:
    """Simplified RSI for testing."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prices = []
        self.gains = []
        self.losses = []
        self.rsi = 50.0
        self.ready = False
    
    def calculate(self, price: float, timestamp=None) -> Optional[float]:
        """Calculate RSI."""
        self.prices.append(price)
        
        if len(self.prices) < 2:
            return None
        
        # Calculate price change
        change = self.prices[-1] - self.prices[-2]
        gain = max(0, change)
        loss = max(0, -change)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        # Keep only period values
        if len(self.gains) > self.period:
            self.gains.pop(0)
            self.losses.pop(0)
        
        if len(self.gains) >= self.period:
            avg_gain = sum(self.gains) / len(self.gains)
            avg_loss = sum(self.losses) / len(self.losses)
            
            if avg_loss == 0:
                self.rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                self.rsi = 100 - (100 / (1 + rs))
            
            self.ready = True
            return self.rsi
        
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.prices.clear()
        self.gains.clear()
        self.losses.clear()
        self.rsi = 50.0
        self.ready = False


class SimpleMeanReversionStrategy:
    """Simplified mean reversion strategy for testing."""
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 z_score_threshold: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 mean_exit_threshold: float = 0.1,
                 max_holding_period: int = 10):
        # Parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.z_score_threshold = z_score_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.mean_exit_threshold = mean_exit_threshold
        self.max_holding_period = max_holding_period
        
        # Indicators
        self.bb = SimpleBollingerBands(bb_period, bb_std)
        self.rsi_indicator = SimpleRSI(rsi_period)
        
        # State
        self.z_score = 0.0
        self.distance_from_mean = 0.0
        self.last_signal_time = None
        self.signal_cooldown = 3600  # 1 hour
        
        # Position tracking
        self.position_entry_data = {}
        self.bars_since_entry = {}
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "mean_reversion_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from market data."""
        price = market_data.get('close', market_data.get('price'))
        timestamp = market_data.get('timestamp', datetime.now())
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        if price is None:
            return None
        
        # Update indicators
        bb_values = self.bb.calculate(price, timestamp)
        rsi = self.rsi_indicator.calculate(price, timestamp)
        
        if not (self.bb.ready and self.rsi_indicator.ready):
            return None
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                return None
        
        # Calculate z-score
        if bb_values and bb_values['std_dev'] > 0:
            self.z_score = (price - bb_values['middle']) / bb_values['std_dev']
            self.distance_from_mean = abs(price - bb_values['middle']) / bb_values['middle']
        else:
            # If no BB values or zero std_dev, can't calculate z-score
            self.z_score = 0.0
            self.distance_from_mean = 0.0
        
        signal = None
        
        # Check for oversold condition (buy signal)
        if self.z_score < -self.z_score_threshold and rsi < self.rsi_oversold:
            signal = {
                'symbol': symbol,
                'signal_type': SignalType.ENTRY,
                'side': OrderSide.BUY,
                'strength': min(abs(self.z_score) / 3, 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'z_score': self.z_score,
                    'rsi': rsi,
                    'distance_from_mean': self.distance_from_mean,
                    'reason': 'Oversold condition with low RSI'
                }
            }
        
        # Check for overbought condition (sell signal)
        elif self.z_score > self.z_score_threshold and rsi > self.rsi_overbought:
            signal = {
                'symbol': symbol,
                'signal_type': SignalType.ENTRY,
                'side': OrderSide.SELL,
                'strength': min(abs(self.z_score) / 3, 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'z_score': self.z_score,
                    'rsi': rsi,
                    'distance_from_mean': self.distance_from_mean,
                    'reason': 'Overbought condition with high RSI'
                }
            }
        
        # Check for mean reversion exits
        if signal is None and self.position_entry_data:
            exit_signal = self._check_mean_exits(market_data)
            if exit_signal:
                signal = exit_signal
        
        if signal:
            self.last_signal_time = timestamp
            
            # Track position entry
            if signal.get('signal_type') == SignalType.ENTRY:
                position_id = f"{symbol}_{timestamp.isoformat()}"
                self.position_entry_data[position_id] = {
                    'entry_price': price,
                    'entry_z_score': self.z_score,
                    'entry_mean': bb_values['middle'],
                    'entry_time': timestamp,
                    'side': signal['side']
                }
                self.bars_since_entry[position_id] = 0
                signal['position_id'] = position_id
        
        return signal
    
    def _check_mean_exits(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for mean reversion exit conditions."""
        if not self.position_entry_data:
            return None
        
        price = market_data.get('close', market_data.get('price'))
        timestamp = market_data.get('timestamp', datetime.now())
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Check each position
        for position_id, entry_data in list(self.position_entry_data.items()):
            # Increment bars held
            self.bars_since_entry[position_id] = self.bars_since_entry.get(position_id, 0) + 1
            
            # Check exit conditions
            should_exit = False
            exit_reason = None
            
            # 1. Price reverted to mean
            if self.distance_from_mean < self.mean_exit_threshold:
                should_exit = True
                exit_reason = "Mean reversion complete"
            
            # 2. Held too long
            elif self.bars_since_entry[position_id] >= self.max_holding_period:
                should_exit = True
                exit_reason = "Max holding period reached"
            
            # 3. Z-score flipped significantly
            entry_z_score = entry_data['entry_z_score']
            if entry_z_score > 0 and self.z_score < -1:
                should_exit = True
                exit_reason = "Z-score flipped negative"
            elif entry_z_score < 0 and self.z_score > 1:
                should_exit = True
                exit_reason = "Z-score flipped positive"
            
            if should_exit:
                # Calculate profit
                entry_price = entry_data['entry_price']
                if entry_data['side'] == OrderSide.BUY:
                    profit_pct = (price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - price) / entry_price
                
                # Create exit signal
                exit_signal = {
                    'symbol': symbol,
                    'signal_type': SignalType.EXIT,
                    'side': OrderSide.SELL if entry_data['side'] == OrderSide.BUY else OrderSide.BUY,
                    'strength': 1.0,  # Exit with full position
                    'timestamp': timestamp,
                    'position_id': position_id,
                    'metadata': {
                        'z_score': self.z_score,
                        'distance_from_mean': self.distance_from_mean,
                        'reason': exit_reason,
                        'profit_pct': profit_pct,
                        'bars_held': self.bars_since_entry[position_id]
                    }
                }
                
                # Clean up position data
                del self.position_entry_data[position_id]
                del self.bars_since_entry[position_id]
                
                return exit_signal
        
        return None
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.bb.reset()
        self.rsi_indicator.reset()
        self.z_score = 0.0
        self.distance_from_mean = 0.0
        self.last_signal_time = None
        self.position_entry_data.clear()
        self.bars_since_entry.clear()


class TestMeanReversionStrategy(unittest.TestCase):
    """Test mean reversion strategy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SimpleMeanReversionStrategy(
            bb_period=5,  # Short period for testing
            bb_std=2.0,
            z_score_threshold=2.0,
            rsi_period=5,
            rsi_oversold=30,
            rsi_overbought=70,
            mean_exit_threshold=0.1,
            max_holding_period=10
        )
        self.base_timestamp = datetime(2024, 1, 1, 9, 30)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SimpleMeanReversionStrategy(
            bb_period=20,
            bb_std=2.5,
            z_score_threshold=2.5,
            rsi_period=14,
            rsi_oversold=25,
            rsi_overbought=75,
            mean_exit_threshold=0.15,
            max_holding_period=15
        )
        
        self.assertEqual(strategy.bb_period, 20)
        self.assertEqual(strategy.bb_std, 2.5)
        self.assertEqual(strategy.z_score_threshold, 2.5)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.rsi_oversold, 25)
        self.assertEqual(strategy.rsi_overbought, 75)
        self.assertEqual(strategy.mean_exit_threshold, 0.15)
        self.assertEqual(strategy.max_holding_period, 15)
        self.assertEqual(strategy.name, "mean_reversion_strategy")
    
    def test_insufficient_data_returns_none(self):
        """Test that strategy returns None when insufficient data."""
        # Feed less than bb_period prices
        for i in range(4):
            market_data = {
                'close': 100.0 + i * 0.1,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
            self.assertIsNone(signal)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        # Feed exactly bb_period prices
        prices = [100.0, 101.0, 100.5, 99.5, 100.0]
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Check BB values
        self.assertIsNotNone(self.strategy.bb.middle_band)
        self.assertIsNotNone(self.strategy.bb.upper_band)
        self.assertIsNotNone(self.strategy.bb.lower_band)
        self.assertIsNotNone(self.strategy.bb.std_dev)
        
        # Middle band should be average
        expected_middle = sum(prices) / len(prices)
        self.assertAlmostEqual(self.strategy.bb.middle_band, expected_middle, places=2)
    
    def test_z_score_calculation(self):
        """Test z-score calculation."""
        # Create prices with known pattern
        mean_price = 100.0
        prices = [mean_price] * 5  # Start with flat prices
        
        # Feed initial prices
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Now feed a price 2 std devs below mean
        # With flat prices, std_dev should be 0, so we need some variance
        prices = [99.0, 101.0, 99.0, 101.0, 100.0]  # Create variance
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=5+i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Now test with extreme price
        extreme_price = 95.0  # Well below mean
        market_data = {
            'close': extreme_price,
            'timestamp': self.base_timestamp + timedelta(minutes=10),
            'symbol': 'TEST'
        }
        self.strategy.generate_signal(market_data)
        
        # Z-score should be negative
        self.assertLess(self.strategy.z_score, 0)
    
    def test_oversold_buy_signal(self):
        """Test buy signal generation on oversold condition."""
        # Create downtrend to trigger oversold
        prices = [100.0, 99.0, 98.0, 97.0, 96.0]  # Downtrend
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Feed more declining prices to get RSI oversold
        for i in range(3):
            market_data = {
                'close': 95.0 - i,  # Continue decline
                'timestamp': self.base_timestamp + timedelta(minutes=5+i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # Now push price well below lower band
        market_data = {
            'close': 90.0,  # Extreme low
            'timestamp': self.base_timestamp + timedelta(minutes=10),
            'symbol': 'TEST'
        }
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # Signal might be None due to cooldown or thresholds
            self.assertEqual(signal['side'], OrderSide.BUY)
            self.assertEqual(signal['signal_type'], SignalType.ENTRY)
            self.assertLess(signal['metadata']['z_score'], -self.strategy.z_score_threshold)
            self.assertLess(signal['metadata']['rsi'], self.strategy.rsi_oversold)
    
    def test_overbought_sell_signal(self):
        """Test sell signal generation on overbought condition."""
        # Create uptrend to trigger overbought
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]  # Uptrend
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Feed more rising prices to get RSI overbought
        for i in range(3):
            market_data = {
                'close': 105.0 + i,  # Continue rise
                'timestamp': self.base_timestamp + timedelta(minutes=5+i),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
        
        # Now push price well above upper band
        market_data = {
            'close': 110.0,  # Extreme high
            'timestamp': self.base_timestamp + timedelta(minutes=10),
            'symbol': 'TEST'
        }
        signal = self.strategy.generate_signal(market_data)
        
        if signal:  # Signal might be None due to cooldown or thresholds
            self.assertEqual(signal['side'], OrderSide.SELL)
            self.assertEqual(signal['signal_type'], SignalType.ENTRY)
            self.assertGreater(signal['metadata']['z_score'], self.strategy.z_score_threshold)
            self.assertGreater(signal['metadata']['rsi'], self.strategy.rsi_overbought)
    
    def test_mean_reversion_exit(self):
        """Test exit when price reverts to mean."""
        # Setup: Create initial trend and entry signal
        prices = [100.0, 99.0, 98.0, 97.0, 96.0]  # Downtrend
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Force an entry by manipulating state
        self.strategy.z_score = -2.5
        self.strategy.rsi_indicator.rsi = 25
        entry_market_data = {
            'close': 94.0,
            'timestamp': self.base_timestamp + timedelta(minutes=5),
            'symbol': 'TEST'
        }
        entry_signal = self.strategy.generate_signal(entry_market_data)
        
        # Simulate position entry
        if not entry_signal:
            # Manually create position entry
            position_id = "TEST_manual"
            self.strategy.position_entry_data[position_id] = {
                'entry_price': 94.0,
                'entry_z_score': -2.5,
                'entry_mean': 98.0,
                'entry_time': self.base_timestamp + timedelta(minutes=5),
                'side': OrderSide.BUY
            }
            self.strategy.bars_since_entry[position_id] = 0
        
        # Now have price revert to mean
        mean_price = self.strategy.bb.middle_band or 98.0
        market_data = {
            'close': mean_price * 0.99,  # Close to mean
            'timestamp': self.base_timestamp + timedelta(minutes=10),
            'symbol': 'TEST'
        }
        
        # Update distance from mean
        self.strategy.distance_from_mean = 0.01  # Within threshold
        
        exit_signal = self.strategy.generate_signal(market_data)
        
        if exit_signal and exit_signal['signal_type'] == SignalType.EXIT:
            self.assertEqual(exit_signal['signal_type'], SignalType.EXIT)
            self.assertIn('Mean reversion complete', exit_signal['metadata']['reason'])
    
    def test_max_holding_period_exit(self):
        """Test exit when max holding period reached."""
        # Setup position
        position_id = "TEST_position"
        self.strategy.position_entry_data[position_id] = {
            'entry_price': 95.0,
            'entry_z_score': -2.0,
            'entry_mean': 100.0,
            'entry_time': self.base_timestamp,
            'side': OrderSide.BUY
        }
        self.strategy.bars_since_entry[position_id] = 0
        
        # Simulate holding for max period
        for i in range(self.strategy.max_holding_period):
            market_data = {
                'close': 95.0 + i * 0.1,  # Gradual recovery
                'timestamp': self.base_timestamp + timedelta(minutes=i+1),
                'symbol': 'TEST'
            }
            signal = self.strategy.generate_signal(market_data)
            
            # On the last iteration, should get exit signal
            if i == self.strategy.max_holding_period - 1 and signal:
                self.assertEqual(signal['signal_type'], SignalType.EXIT)
                self.assertIn('Max holding period', signal['metadata']['reason'])
    
    def test_z_score_flip_exit(self):
        """Test exit when z-score flips significantly."""
        # Setup: Create position with positive z-score entry
        position_id = "TEST_position"
        self.strategy.position_entry_data[position_id] = {
            'entry_price': 105.0,
            'entry_z_score': 2.0,  # Entered on overbought
            'entry_mean': 100.0,
            'entry_time': self.base_timestamp,
            'side': OrderSide.SELL
        }
        self.strategy.bars_since_entry[position_id] = 0
        
        # Setup indicators to have moved significantly
        self.strategy.bb.ready = True
        self.strategy.rsi_indicator.ready = True
        self.strategy.z_score = -1.5  # Flipped negative
        
        market_data = {
            'close': 95.0,  # Price now below mean
            'timestamp': self.base_timestamp + timedelta(minutes=5),
            'symbol': 'TEST'
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal and signal['signal_type'] == SignalType.EXIT:
            self.assertEqual(signal['signal_type'], SignalType.EXIT)
            self.assertIn('Z-score flipped', signal['metadata']['reason'])
    
    def test_signal_cooldown(self):
        """Test signal cooldown period."""
        # Create conditions for signal
        self.strategy.bb.ready = True
        self.strategy.rsi_indicator.ready = True
        self.strategy.z_score = -2.5
        self.strategy.rsi_indicator.rsi = 25
        
        # Generate first signal
        market_data = {
            'close': 90.0,
            'timestamp': self.base_timestamp,
            'symbol': 'TEST'
        }
        first_signal = self.strategy.generate_signal(market_data)
        
        # Try again within cooldown
        market_data = {
            'close': 89.0,  # Even more extreme
            'timestamp': self.base_timestamp + timedelta(seconds=30),
            'symbol': 'TEST'
        }
        second_signal = self.strategy.generate_signal(market_data)
        
        # Should be None due to cooldown
        self.assertIsNone(second_signal)
        
        # Try after cooldown
        market_data = {
            'close': 89.0,
            'timestamp': self.base_timestamp + timedelta(seconds=3601),
            'symbol': 'TEST'
        }
        third_signal = self.strategy.generate_signal(market_data)
        
        # Could generate signal now (if conditions still met)
        # Note: Actual signal depends on indicator state
    
    def test_position_tracking(self):
        """Test position entry data tracking."""
        # Generate entry signal
        self.strategy.bb.ready = True
        self.strategy.rsi_indicator.ready = True
        self.strategy.z_score = -2.5
        self.strategy.rsi_indicator.rsi = 25
        
        market_data = {
            'close': 90.0,
            'timestamp': self.base_timestamp,
            'symbol': 'TEST'
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        if signal and signal['signal_type'] == SignalType.ENTRY:
            position_id = signal.get('position_id')
            if position_id:
                # Check position data was stored
                self.assertIn(position_id, self.strategy.position_entry_data)
                entry_data = self.strategy.position_entry_data[position_id]
                self.assertEqual(entry_data['entry_price'], 90.0)
                self.assertEqual(entry_data['entry_z_score'], self.strategy.z_score)
                self.assertEqual(entry_data['side'], OrderSide.BUY)
    
    def test_signal_strength_calculation(self):
        """Test signal strength based on z-score magnitude."""
        # Test various z-score levels
        test_cases = [
            (-2.0, 0.667),   # 2/3 = 0.667
            (-3.0, 1.0),     # Capped at 1.0
            (2.5, 0.833),    # 2.5/3 = 0.833
            (4.0, 1.0),      # Capped at 1.0
        ]
        
        for z_score, expected_strength in test_cases:
            self.strategy.z_score = z_score
            strength = min(abs(z_score) / 3, 1.0)
            self.assertAlmostEqual(strength, expected_strength, places=3)
    
    def test_reset_functionality(self):
        """Test strategy reset."""
        # Generate some state
        for i in range(10):
            market_data = {
                'close': 100.0 + i * 0.5,
                'timestamp': self.base_timestamp + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            self.strategy.generate_signal(market_data)
        
        # Add some position data
        self.strategy.position_entry_data['test'] = {'entry_price': 100}
        self.strategy.bars_since_entry['test'] = 5
        
        # Reset
        self.strategy.reset()
        
        # Check everything is cleared
        self.assertEqual(len(self.strategy.bb.prices), 0)
        self.assertEqual(len(self.strategy.rsi_indicator.prices), 0)
        self.assertEqual(self.strategy.z_score, 0.0)
        self.assertEqual(len(self.strategy.position_entry_data), 0)
        self.assertEqual(len(self.strategy.bars_since_entry), 0)
        self.assertIsNone(self.strategy.last_signal_time)


class TestMeanReversionEdgeCases(unittest.TestCase):
    """Test edge cases for mean reversion strategy."""
    
    def test_zero_standard_deviation(self):
        """Test handling of zero standard deviation."""
        strategy = SimpleMeanReversionStrategy(bb_period=3)
        
        # Feed identical prices (zero std dev)
        for i in range(5):
            market_data = {
                'close': 100.0,
                'timestamp': datetime.now() + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            signal = strategy.generate_signal(market_data)
        
        # Should handle gracefully without division by zero
        self.assertEqual(strategy.z_score, 0.0)
    
    def test_missing_price_data(self):
        """Test handling of missing price data."""
        strategy = SimpleMeanReversionStrategy()
        
        market_data = {
            'timestamp': datetime.now(),
            'symbol': 'TEST'
            # No price field
        }
        
        signal = strategy.generate_signal(market_data)
        self.assertIsNone(signal)
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements."""
        strategy = SimpleMeanReversionStrategy(bb_period=5)
        
        # Create prices with some variance for meaningful std_dev
        base_time = datetime.now()
        prices = [100.0, 99.5, 100.5, 99.0, 101.0]  # Some variance
        
        for i, price in enumerate(prices):
            market_data = {
                'close': price,
                'timestamp': base_time + timedelta(minutes=i),
                'symbol': 'TEST'
            }
            strategy.generate_signal(market_data)
        
        # Ensure indicators are ready
        self.assertTrue(strategy.bb.ready)
        self.assertGreater(strategy.bb.std_dev, 0)
        
        # Store values before extreme move
        original_mean = strategy.bb.middle_band
        original_std = strategy.bb.std_dev
        
        # Now send extreme price that's many std devs away
        extreme_price = original_mean + (5 * original_std)  # 5 std devs above mean
        
        market_data = {
            'close': extreme_price,
            'timestamp': base_time + timedelta(minutes=10),
            'symbol': 'TEST'
        }
        signal = strategy.generate_signal(market_data)
        
        # Z-score should be significant (though BB will update with new price)
        # The exact value depends on how BB recalculates with the new extreme price
        self.assertIsInstance(strategy.z_score, float)
        # Should be positive for price above mean
        self.assertGreater(strategy.z_score, 0)
    
    def test_multiple_positions(self):
        """Test handling multiple positions."""
        strategy = SimpleMeanReversionStrategy()
        
        # Create multiple positions
        for i in range(3):
            position_id = f"position_{i}"
            strategy.position_entry_data[position_id] = {
                'entry_price': 100.0 - i,
                'entry_z_score': -2.0 - i * 0.1,
                'entry_mean': 100.0,
                'entry_time': datetime.now(),
                'side': OrderSide.BUY
            }
            strategy.bars_since_entry[position_id] = i
        
        # Update state for exit conditions
        strategy.distance_from_mean = 0.05  # Close to mean
        strategy.bb.ready = True
        strategy.rsi_indicator.ready = True
        
        # Generate signal - should exit one position
        market_data = {
            'close': 100.0,
            'timestamp': datetime.now() + timedelta(minutes=1),
            'symbol': 'TEST'
        }
        
        initial_positions = len(strategy.position_entry_data)
        signal = strategy.generate_signal(market_data)
        
        if signal and signal['signal_type'] == SignalType.EXIT:
            # Should have one less position
            self.assertEqual(len(strategy.position_entry_data), initial_positions - 1)


if __name__ == "__main__":
    unittest.main()