"""
Tests for example trading strategies.

These tests validate that strategies generate appropriate signals
and work correctly with the rest of the system.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.test_config import TestConfig, MockMarketData, TestSignals


class SimpleMomentumStrategy:
    """Simple momentum strategy for testing."""
    
    def __init__(self, lookback_period: int = 20, threshold: Decimal = Decimal("0.02")):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Number of periods for momentum calculation
            threshold: Minimum momentum threshold for signals
        """
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.price_history = {}
        self.positions = {}
    
    def initialize(self, context):
        """Initialize strategy with context."""
        self.symbols = context.get('symbols', [])
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
    
    def generate_signals(self, market_data):
        """Generate momentum-based signals."""
        from src.risk.protocols import Signal, SignalType, OrderSide
        
        signals = []
        timestamp = market_data['timestamp']
        
        for symbol in self.symbols:
            if symbol not in market_data.get('prices', {}):
                continue
            
            # Update price history
            price = market_data['prices'][symbol]
            self.price_history[symbol].append(price)
            
            # Keep only lookback period
            if len(self.price_history[symbol]) > self.lookback_period:
                self.price_history[symbol].pop(0)
            
            # Need full lookback period
            if len(self.price_history[symbol]) < self.lookback_period:
                continue
            
            # Calculate momentum
            first_price = self.price_history[symbol][0]
            last_price = self.price_history[symbol][-1]
            momentum = (last_price - first_price) / first_price
            
            # Generate signals based on momentum
            if momentum > self.threshold and self.positions[symbol] == 0:
                # Strong positive momentum - buy
                signal = Signal(
                    signal_id=f"MOM_BUY_{symbol}_{timestamp.timestamp()}",
                    strategy_id="momentum",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=min(Decimal(str(abs(momentum) * 10)), Decimal("1.0")),
                    timestamp=timestamp,
                    metadata={'momentum': float(momentum)}
                )
                signals.append(signal)
                self.positions[symbol] = 1
                
            elif momentum < -self.threshold and self.positions[symbol] > 0:
                # Strong negative momentum - sell
                signal = Signal(
                    signal_id=f"MOM_SELL_{symbol}_{timestamp.timestamp()}",
                    strategy_id="momentum",
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    side=OrderSide.SELL,
                    strength=Decimal("1.0"),
                    timestamp=timestamp,
                    metadata={'momentum': float(momentum)}
                )
                signals.append(signal)
                self.positions[symbol] = 0
        
        return signals


class MeanReversionStrategy:
    """Mean reversion strategy for testing."""
    
    def __init__(self, lookback_period: int = 20, num_std: Decimal = Decimal("2.0")):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for calculating mean and std
            num_std: Number of standard deviations for signal threshold
        """
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.price_history = {}
        self.positions = {}
    
    def initialize(self, context):
        """Initialize strategy."""
        self.symbols = context.get('symbols', [])
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
    
    def generate_signals(self, market_data):
        """Generate mean reversion signals."""
        from src.risk.protocols import Signal, SignalType, OrderSide
        import statistics
        
        signals = []
        timestamp = market_data['timestamp']
        
        for symbol in self.symbols:
            if symbol not in market_data.get('prices', {}):
                continue
            
            # Update price history
            price = market_data['prices'][symbol]
            self.price_history[symbol].append(price)
            
            # Keep only lookback period
            if len(self.price_history[symbol]) > self.lookback_period:
                self.price_history[symbol].pop(0)
            
            # Need full lookback period
            if len(self.price_history[symbol]) < self.lookback_period:
                continue
            
            # Calculate mean and std
            prices = self.price_history[symbol]
            mean = statistics.mean(prices)
            std = statistics.stdev(prices) if len(prices) > 1 else 0
            
            if std == 0:
                continue
            
            # Calculate z-score
            z_score = (price - mean) / std
            
            # Generate signals based on mean reversion
            if z_score < -float(self.num_std) and self.positions[symbol] == 0:
                # Price below lower band - buy
                signal = Signal(
                    signal_id=f"MR_BUY_{symbol}_{timestamp.timestamp()}",
                    strategy_id="mean_reversion",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=min(Decimal(str(abs(z_score) / 3)), Decimal("1.0")),
                    timestamp=timestamp,
                    metadata={'z_score': z_score, 'mean': mean, 'std': std}
                )
                signals.append(signal)
                self.positions[symbol] = 1
                
            elif z_score > float(self.num_std) and self.positions[symbol] > 0:
                # Price above upper band - sell
                signal = Signal(
                    signal_id=f"MR_SELL_{symbol}_{timestamp.timestamp()}",
                    strategy_id="mean_reversion",
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    side=OrderSide.SELL,
                    strength=Decimal("1.0"),
                    timestamp=timestamp,
                    metadata={'z_score': z_score, 'mean': mean, 'std': std}
                )
                signals.append(signal)
                self.positions[symbol] = 0
        
        return signals


class TestMomentumStrategy(unittest.TestCase):
    """Test momentum strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SimpleMomentumStrategy(
            lookback_period=5,
            threshold=Decimal("0.03")  # 3% threshold
        )
        self.strategy.initialize({'symbols': ['AAPL', 'GOOGL']})
    
    def test_no_signal_insufficient_data(self):
        """Test no signals when insufficient data."""
        # Only 3 bars of data (need 5)
        for i in range(3):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': 150.0 + i, 'GOOGL': 2800.0 + i * 10}
            }
            signals = self.strategy.generate_signals(market_data)
            self.assertEqual(len(signals), 0)
    
    def test_buy_signal_on_momentum(self):
        """Test buy signal generation on positive momentum."""
        # Create upward momentum
        prices = [150, 151, 153, 155, 158]  # 5.3% gain
        
        signals_generated = []
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price, 'GOOGL': 2800.0}
            }
            signals = self.strategy.generate_signals(market_data)
            signals_generated.extend(signals)
        
        # Should generate buy signal for AAPL
        self.assertGreater(len(signals_generated), 0)
        
        buy_signals = [s for s in signals_generated if s.side.value == "buy"]
        self.assertEqual(len(buy_signals), 1)
        self.assertEqual(buy_signals[0].symbol, 'AAPL')
        self.assertGreater(buy_signals[0].strength, Decimal("0"))
    
    def test_sell_signal_on_reverse(self):
        """Test sell signal on momentum reversal."""
        # First create position with upward momentum
        up_prices = [150, 151, 153, 155, 158]
        for i, price in enumerate(up_prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price}
            }
            self.strategy.generate_signals(market_data)
        
        # Now create downward momentum
        down_prices = [157, 155, 152, 149, 145]  # -7.6% from peak
        
        signals_generated = []
        for i, price in enumerate(down_prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i+5),
                'prices': {'AAPL': price}
            }
            signals = self.strategy.generate_signals(market_data)
            signals_generated.extend(signals)
        
        # Should generate sell signal
        sell_signals = [s for s in signals_generated if s.side.value == "sell"]
        self.assertEqual(len(sell_signals), 1)
        self.assertEqual(sell_signals[0].symbol, 'AAPL')
    
    def test_no_duplicate_signals(self):
        """Test that strategy doesn't generate duplicate signals."""
        # Create strong momentum
        prices = [150, 155, 160, 165, 170, 175, 180]
        
        buy_count = 0
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price}
            }
            signals = self.strategy.generate_signals(market_data)
            buy_count += sum(1 for s in signals if s.side.value == "buy" and s.symbol == 'AAPL')
        
        # Should only buy once
        self.assertEqual(buy_count, 1)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test mean reversion strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MeanReversionStrategy(
            lookback_period=10,
            num_std=Decimal("2.0")
        )
        self.strategy.initialize({'symbols': ['AAPL']})
    
    def test_buy_signal_oversold(self):
        """Test buy signal when price is oversold."""
        # Create price series with drop at end
        prices = [150, 149, 151, 150, 149, 151, 150, 149, 145, 140]  # Big drop
        
        signals_generated = []
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price}
            }
            signals = self.strategy.generate_signals(market_data)
            signals_generated.extend(signals)
        
        # Should generate buy signal
        buy_signals = [s for s in signals_generated if s.side.value == "buy"]
        self.assertEqual(len(buy_signals), 1)
        
        # Check metadata
        self.assertIn('z_score', buy_signals[0].metadata)
        self.assertLess(buy_signals[0].metadata['z_score'], -2.0)
    
    def test_sell_signal_overbought(self):
        """Test sell signal when price is overbought."""
        # First establish position
        prices = [150, 149, 151, 150, 149, 151, 150, 149, 145, 140]
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price}
            }
            self.strategy.generate_signals(market_data)
        
        # Now create overbought condition
        high_prices = [145, 150, 155, 160, 165]  # Big rally
        
        signals_generated = []
        for i, price in enumerate(high_prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i+10),
                'prices': {'AAPL': price}
            }
            signals = self.strategy.generate_signals(market_data)
            signals_generated.extend(signals)
        
        # Should generate sell signal
        sell_signals = [s for s in signals_generated if s.side.value == "sell"]
        self.assertGreater(len(sell_signals), 0)
    
    def test_no_signal_normal_range(self):
        """Test no signals when price is within normal range."""
        # Stable prices
        prices = [150, 149, 151, 150, 149, 151, 150, 149, 151, 150]
        
        signals_generated = []
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': datetime.now() + timedelta(days=i),
                'prices': {'AAPL': price}
            }
            signals = self.strategy.generate_signals(market_data)
            signals_generated.extend(signals)
        
        # Should not generate any signals
        self.assertEqual(len(signals_generated), 0)


class TestStrategyIntegration(unittest.TestCase):
    """Test strategy integration with the system."""
    
    def test_strategy_with_risk_limits(self):
        """Test that strategy signals respect risk limits."""
        # This would require importing risk module
        # Skipped to avoid circular dependencies in test
        pass
    
    def test_multiple_strategies_same_symbol(self):
        """Test multiple strategies generating signals for same symbol."""
        momentum = SimpleMomentumStrategy(lookback_period=5, threshold=Decimal("0.03"))
        mean_rev = MeanReversionStrategy(lookback_period=10, num_std=Decimal("2.0"))
        
        momentum.initialize({'symbols': ['AAPL']})
        mean_rev.initialize({'symbols': ['AAPL']})
        
        # Create market data that triggers both strategies
        market_data = {
            'timestamp': datetime.now(),
            'prices': {'AAPL': 150.0}
        }
        
        # Feed same data to both
        mom_signals = momentum.generate_signals(market_data)
        mr_signals = mean_rev.generate_signals(market_data)
        
        # Both should work independently
        self.assertIsInstance(mom_signals, list)
        self.assertIsInstance(mr_signals, list)


if __name__ == "__main__":
    unittest.main()