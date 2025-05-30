"""
Tests for advanced signal processing components.

These tests validate the signal router, validator, cache, and prioritizer.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.risk.protocols import (
    Signal,
    SignalType,
    OrderSide,
    Order,
    OrderType,
    PortfolioStateProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
    SignalProcessorProtocol
)
from src.risk.signal_advanced import (
    SignalRouter,
    SignalValidator,
    RiskAdjustedSignalProcessor,
    SignalCache,
    SignalPrioritizer,
    SignalPriority
)


class TestSignalRouter(unittest.TestCase):
    """Test signal router functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = SignalRouter()
        self.mock_processor = Mock(spec=SignalProcessorProtocol)
        self.mock_processor.process_signal.return_value = Mock(spec=Order)
        
        # Mock portfolio and dependencies
        self.mock_portfolio = Mock(spec=PortfolioStateProtocol)
        self.mock_sizer = Mock(spec=PositionSizerProtocol)
        self.mock_limits = []
        self.market_data = {"prices": {"AAPL": 150.0}}
    
    def test_default_processor(self):
        """Test routing to default processor."""
        self.router.set_default_processor(self.mock_processor)
        
        signal = Signal(
            signal_id="SIG001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        order = self.router.route_signal(
            signal,
            self.mock_portfolio,
            self.mock_sizer,
            self.mock_limits,
            self.market_data
        )
        
        self.assertIsNotNone(order)
        self.mock_processor.process_signal.assert_called_once()
    
    def test_strategy_specific_processor(self):
        """Test routing to strategy-specific processor."""
        strategy_processor = Mock(spec=SignalProcessorProtocol)
        strategy_processor.process_signal.return_value = Mock(spec=Order)
        
        self.router.set_default_processor(self.mock_processor)
        self.router.add_strategy_processor("momentum", strategy_processor)
        
        signal = Signal(
            signal_id="SIG002",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        order = self.router.route_signal(
            signal,
            self.mock_portfolio,
            self.mock_sizer,
            self.mock_limits,
            self.market_data
        )
        
        # Should use strategy processor, not default
        strategy_processor.process_signal.assert_called_once()
        self.mock_processor.process_signal.assert_not_called()
    
    def test_signal_type_processor(self):
        """Test routing to signal type processor."""
        exit_processor = Mock(spec=SignalProcessorProtocol)
        exit_processor.process_signal.return_value = Mock(spec=Order)
        
        self.router.set_default_processor(self.mock_processor)
        self.router.add_signal_type_processor(SignalType.EXIT, exit_processor)
        
        signal = Signal(
            signal_id="SIG003",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            side=OrderSide.SELL,
            strength=Decimal("1.0"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        order = self.router.route_signal(
            signal,
            self.mock_portfolio,
            self.mock_sizer,
            self.mock_limits,
            self.market_data
        )
        
        # Should use exit processor
        exit_processor.process_signal.assert_called_once()
        self.mock_processor.process_signal.assert_not_called()
    
    def test_no_processor_found(self):
        """Test behavior when no processor is found."""
        signal = Signal(
            signal_id="SIG004",
            strategy_id="unknown",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        order = self.router.route_signal(
            signal,
            self.mock_portfolio,
            self.mock_sizer,
            self.mock_limits,
            self.market_data
        )
        
        self.assertIsNone(order)


class TestSignalValidator(unittest.TestCase):
    """Test signal validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SignalValidator(max_signal_age=300)
    
    def test_valid_signal(self):
        """Test validation of valid signal."""
        signal = Signal(
            signal_id="SIG001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        is_valid, failures = self.validator.validate(signal)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(failures), 0)
    
    def test_invalid_strength(self):
        """Test signal with invalid strength."""
        signal = Signal(
            signal_id="SIG002",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.5"),  # Out of range
            timestamp=datetime.now(),
            metadata={}
        )
        
        is_valid, failures = self.validator.validate(signal)
        
        self.assertFalse(is_valid)
        self.assertIn("strength_range", failures[0])
    
    def test_missing_required_fields(self):
        """Test signal with missing required fields."""
        signal = Signal(
            signal_id="",  # Empty
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        is_valid, failures = self.validator.validate(signal)
        
        self.assertFalse(is_valid)
        self.assertIn("required_fields", failures[0])
    
    def test_old_signal(self):
        """Test signal that is too old."""
        signal = Signal(
            signal_id="SIG003",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now() - timedelta(seconds=400),  # Too old
            metadata={}
        )
        
        is_valid, failures = self.validator.validate(signal)
        
        self.assertFalse(is_valid)
        self.assertIn("timestamp_valid", failures[0])
    
    def test_custom_validation_rule(self):
        """Test adding custom validation rule."""
        def check_symbol_length(signal):
            if len(signal.symbol) > 5:
                return False, "Symbol too long"
            return True, None
        
        self.validator.add_rule("symbol_length", check_symbol_length)
        
        signal = Signal(
            signal_id="SIG004",
            strategy_id="test_strategy",
            symbol="VERYLONGSYMBOL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        is_valid, failures = self.validator.validate(signal)
        
        self.assertFalse(is_valid)
        self.assertIn("symbol_length", failures[0])


class TestRiskAdjustedSignalProcessor(unittest.TestCase):
    """Test risk-adjusted signal processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = RiskAdjustedSignalProcessor(risk_multiplier=Decimal("0.8"))
        
        # Mock dependencies
        self.mock_portfolio = Mock(spec=PortfolioStateProtocol)
        self.mock_portfolio.get_position.return_value = None
        self.mock_portfolio.get_cash_balance.return_value = Decimal("100000")
        
        self.mock_sizer = Mock(spec=PositionSizerProtocol)
        self.mock_sizer.calculate_size.return_value = Decimal("100")
        
        self.mock_limits = []
        self.market_data = {"prices": {"AAPL": 150.0}}
    
    def test_risk_exit_conversion(self):
        """Test that risk exits are converted to market orders."""
        signal = Signal(
            signal_id="SIG001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.RISK_EXIT,
            side=OrderSide.SELL,
            strength=Decimal("1.0"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        order = self.processor.process_signal(
            signal,
            self.mock_portfolio,
            self.mock_sizer,
            self.mock_limits,
            self.market_data
        )
        
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.time_in_force, "IOC")
        self.assertEqual(order.metadata["priority"], "high")
        self.assertTrue(order.metadata["risk_exit"])


class TestSignalCache(unittest.TestCase):
    """Test signal cache and deduplication."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = SignalCache(cache_duration=60, max_cache_size=100)
    
    def test_duplicate_detection(self):
        """Test duplicate signal detection."""
        signal1 = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # First signal should not be duplicate
        self.assertFalse(self.cache.is_duplicate(signal1))
        self.cache.add_signal(signal1)
        
        # Same signal (different ID but same content) should be duplicate
        signal2 = Signal(
            signal_id="SIG002",  # Different ID
            strategy_id="momentum",  # Same strategy
            symbol="AAPL",  # Same symbol
            signal_type=SignalType.ENTRY,  # Same type
            side=OrderSide.BUY,  # Same side
            strength=Decimal("0.8"),  # Same strength
            timestamp=datetime.now(),
            metadata={}
        )
        
        self.assertTrue(self.cache.is_duplicate(signal2))
    
    def test_cache_expiration(self):
        """Test that cached signals expire."""
        # Create cache with short duration
        cache = SignalCache(cache_duration=1, max_cache_size=100)
        
        signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        self.assertFalse(cache.is_duplicate(signal))
        cache.add_signal(signal)
        
        # Sleep to let cache expire
        import time
        time.sleep(1.1)
        
        # Should no longer be duplicate
        self.assertFalse(cache.is_duplicate(signal))
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = SignalCache(cache_duration=3600, max_cache_size=3)
        
        # Add 4 signals
        for i in range(4):
            signal = Signal(
                signal_id=f"SIG{i:03d}",
                strategy_id="momentum",
                symbol=f"SYM{i}",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            )
            cache.add_signal(signal)
        
        # Cache should only have 3 items
        self.assertEqual(len(cache._cache), 3)
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Miss
        self.cache.is_duplicate(signal)
        self.cache.add_signal(signal)
        
        # Hit
        self.cache.is_duplicate(signal)
        
        stats = self.cache.get_statistics()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["duplicates_rejected"], 1)


class TestSignalPrioritizer(unittest.TestCase):
    """Test signal prioritization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prioritizer = SignalPrioritizer()
    
    def test_signal_type_priority(self):
        """Test prioritization by signal type."""
        signals = [
            Signal(
                signal_id="SIG001",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG002",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.RISK_EXIT,
                side=OrderSide.SELL,
                strength=Decimal("1.0"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG003",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        prioritized = self.prioritizer.prioritize(signals)
        
        # Risk exit should be first
        self.assertEqual(prioritized[0].signal_type, SignalType.RISK_EXIT)
        # Regular exit should be second
        self.assertEqual(prioritized[1].signal_type, SignalType.EXIT)
        # Entry should be last
        self.assertEqual(prioritized[2].signal_type, SignalType.ENTRY)
    
    def test_strength_priority(self):
        """Test prioritization by signal strength."""
        signals = [
            Signal(
                signal_id="SIG001",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.5"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG002",
                strategy_id="test",
                symbol="GOOGL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG003",
                strategy_id="test",
                symbol="MSFT",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.7"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        prioritized = self.prioritizer.prioritize(signals)
        
        # Highest strength should be first
        self.assertEqual(prioritized[0].strength, Decimal("0.9"))
        self.assertEqual(prioritized[1].strength, Decimal("0.7"))
        self.assertEqual(prioritized[2].strength, Decimal("0.5"))
    
    def test_custom_priority_rule(self):
        """Test adding custom priority rule."""
        def prioritize_by_symbol(signal):
            # Prioritize AAPL
            if signal.symbol == "AAPL":
                return -10.0  # Very high priority
            return 0.0
        
        self.prioritizer.add_rule("symbol_priority", prioritize_by_symbol)
        
        signals = [
            Signal(
                signal_id="SIG001",
                strategy_id="test",
                symbol="GOOGL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG002",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.5"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        prioritized = self.prioritizer.prioritize(signals)
        
        # AAPL should be first despite lower strength
        self.assertEqual(prioritized[0].symbol, "AAPL")


if __name__ == "__main__":
    unittest.main()