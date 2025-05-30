"""
Tests for signal flow management.

These tests validate the signal flow manager and multi-symbol signal flow.
"""

import unittest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock
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
    RiskLimitProtocol
)
from src.risk.signal_flow import SignalFlowManager, MultiSymbolSignalFlow
from src.core.events.event_bus import EventBus


class TestSignalFlowManager(unittest.TestCase):
    """Test signal flow manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.flow_manager = SignalFlowManager(
            event_bus=self.event_bus,
            enable_caching=True,
            enable_validation=True,
            enable_aggregation=True,
            aggregation_method="weighted_average"
        )
        
        # Mock dependencies
        self.mock_portfolio = Mock(spec=PortfolioStateProtocol)
        self.mock_portfolio.get_position.return_value = None
        self.mock_portfolio.get_cash_balance.return_value = Decimal("100000")
        
        self.mock_sizer = Mock(spec=PositionSizerProtocol)
        self.mock_sizer.calculate_size.return_value = Decimal("100")
        
        self.mock_limits = []
        self.market_data = {"prices": {"AAPL": 150.0, "GOOGL": 2800.0}}
    
    def test_register_strategy(self):
        """Test strategy registration."""
        self.flow_manager.register_strategy("momentum", Decimal("1.5"))
        
        self.assertIn("momentum", self.flow_manager._registered_strategies)
        self.assertEqual(self.flow_manager._strategy_weights["momentum"], Decimal("1.5"))
    
    def test_unregister_strategy(self):
        """Test strategy unregistration."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        self.flow_manager.unregister_strategy("momentum")
        
        self.assertNotIn("momentum", self.flow_manager._registered_strategies)
        self.assertNotIn("momentum", self.flow_manager._strategy_weights)
    
    def test_signal_collection_async(self):
        """Test asynchronous signal collection."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        
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
        
        # Run async test
        async def test():
            await self.flow_manager.collect_signal(signal)
            # Check signal is in buffer
            async with self.flow_manager._buffer_lock:
                self.assertEqual(len(self.flow_manager._signal_buffer), 1)
                self.assertEqual(self.flow_manager._signal_buffer[0].signal_id, "SIG001")
        
        asyncio.run(test())
    
    def test_signal_rejection_unregistered_strategy(self):
        """Test rejection of signals from unregistered strategies."""
        signal = Signal(
            signal_id="SIG001",
            strategy_id="unknown",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        async def test():
            await self.flow_manager.collect_signal(signal)
            # Signal should be rejected
            async with self.flow_manager._buffer_lock:
                self.assertEqual(len(self.flow_manager._signal_buffer), 0)
            self.assertEqual(self.flow_manager._signals_rejected, 1)
        
        asyncio.run(test())
    
    def test_signal_validation(self):
        """Test signal validation during collection."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        
        # Invalid signal (strength out of range)
        signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.5"),  # Invalid
            timestamp=datetime.now(),
            metadata={}
        )
        
        async def test():
            await self.flow_manager.collect_signal(signal)
            # Signal should be rejected
            async with self.flow_manager._buffer_lock:
                self.assertEqual(len(self.flow_manager._signal_buffer), 0)
            self.assertEqual(self.flow_manager._signals_rejected, 1)
        
        asyncio.run(test())
    
    def test_signal_deduplication(self):
        """Test signal deduplication via cache."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        
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
        
        signal2 = Signal(
            signal_id="SIG002",  # Different ID
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),  # Same content
            timestamp=datetime.now(),
            metadata={}
        )
        
        async def test():
            await self.flow_manager.collect_signal(signal1)
            await self.flow_manager.collect_signal(signal2)
            
            # Only first signal should be in buffer
            async with self.flow_manager._buffer_lock:
                self.assertEqual(len(self.flow_manager._signal_buffer), 1)
            self.assertEqual(self.flow_manager._signals_rejected, 1)
        
        asyncio.run(test())
    
    def test_process_signals(self):
        """Test signal processing into orders."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        self.flow_manager.register_strategy("mean_reversion", Decimal("0.5"))
        
        signals = [
            Signal(
                signal_id="SIG001",
                strategy_id="momentum",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG002",
                strategy_id="mean_reversion",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.6"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        async def test():
            # Collect signals
            for signal in signals:
                await self.flow_manager.collect_signal(signal)
            
            # Process signals
            orders = await self.flow_manager.process_signals(
                self.mock_portfolio,
                self.mock_sizer,
                self.mock_limits,
                self.market_data
            )
            
            # Should aggregate signals and generate order
            self.assertGreater(len(orders), 0)
            self.assertEqual(self.flow_manager._total_orders_generated, len(orders))
        
        asyncio.run(test())
    
    def test_order_callbacks(self):
        """Test order generation callbacks."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        
        # Add callback
        callback_called = False
        callback_order = None
        
        def order_callback(order):
            nonlocal callback_called, callback_order
            callback_called = True
            callback_order = order
        
        self.flow_manager.add_order_callback(order_callback)
        
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
        
        async def test():
            await self.flow_manager.collect_signal(signal)
            
            orders = await self.flow_manager.process_signals(
                self.mock_portfolio,
                self.mock_sizer,
                self.mock_limits,
                self.market_data
            )
            
            # Callback should be called
            self.assertTrue(callback_called)
            self.assertIsNotNone(callback_order)
        
        asyncio.run(test())
    
    def test_statistics(self):
        """Test flow statistics."""
        self.flow_manager.register_strategy("momentum", Decimal("1.0"))
        
        # Generate some activity
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
        
        async def test():
            await self.flow_manager.collect_signal(signal)
            await self.flow_manager.process_signals(
                self.mock_portfolio,
                self.mock_sizer,
                self.mock_limits,
                self.market_data
            )
        
        asyncio.run(test())
        
        stats = self.flow_manager.get_statistics()
        self.assertEqual(stats["total_signals_received"], 1)
        self.assertIn("signal_processor", stats)


class TestMultiSymbolSignalFlow(unittest.TestCase):
    """Test multi-symbol signal flow management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.multi_flow = MultiSymbolSignalFlow(event_bus=self.event_bus)
    
    def test_create_flow_manager(self):
        """Test creating flow manager for classifier."""
        config = {
            "enable_caching": True,
            "enable_validation": True,
            "enable_aggregation": False,
            "aggregation_method": "first"
        }
        
        flow_manager = self.multi_flow.create_flow_manager("equity", config)
        
        self.assertIsNotNone(flow_manager)
        self.assertIn("equity", self.multi_flow._flow_managers)
        self.assertFalse(flow_manager.enable_aggregation)
    
    def test_symbol_to_classifier_mapping(self):
        """Test mapping symbols to classifiers."""
        self.multi_flow.map_symbol_to_classifier("AAPL", "equity")
        self.multi_flow.map_symbol_to_classifier("BTC-USD", "crypto")
        
        self.assertEqual(self.multi_flow._symbol_classifiers["AAPL"], "equity")
        self.assertEqual(self.multi_flow._symbol_classifiers["BTC-USD"], "crypto")
    
    def test_signal_routing(self):
        """Test routing signals to appropriate flow managers."""
        # Create flow managers
        equity_flow = self.multi_flow.create_flow_manager("equity", {})
        crypto_flow = self.multi_flow.create_flow_manager("crypto", {})
        
        # Register strategies
        equity_flow.register_strategy("momentum", Decimal("1.0"))
        crypto_flow.register_strategy("momentum", Decimal("1.0"))
        
        # Map symbols
        self.multi_flow.map_symbol_to_classifier("AAPL", "equity")
        self.multi_flow.map_symbol_to_classifier("BTC-USD", "crypto")
        
        # Create signals
        equity_signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        crypto_signal = Signal(
            signal_id="SIG002",
            strategy_id="momentum",
            symbol="BTC-USD",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.9"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        async def test():
            # Route signals
            await self.multi_flow.route_signal(equity_signal)
            await self.multi_flow.route_signal(crypto_signal)
            
            # Check signals went to correct flow managers
            async with equity_flow._buffer_lock:
                self.assertEqual(len(equity_flow._signal_buffer), 1)
                self.assertEqual(equity_flow._signal_buffer[0].symbol, "AAPL")
            
            async with crypto_flow._buffer_lock:
                self.assertEqual(len(crypto_flow._signal_buffer), 1)
                self.assertEqual(crypto_flow._signal_buffer[0].symbol, "BTC-USD")
        
        asyncio.run(test())
    
    def test_process_all_signals(self):
        """Test processing signals for all classifiers."""
        # Create flow managers
        equity_flow = self.multi_flow.create_flow_manager("equity", {})
        crypto_flow = self.multi_flow.create_flow_manager("crypto", {})
        
        equity_flow.register_strategy("momentum", Decimal("1.0"))
        crypto_flow.register_strategy("momentum", Decimal("1.0"))
        
        # Mock dependencies
        mock_portfolios = {
            "equity": Mock(spec=PortfolioStateProtocol),
            "crypto": Mock(spec=PortfolioStateProtocol)
        }
        for portfolio in mock_portfolios.values():
            portfolio.get_position.return_value = None
            portfolio.get_cash_balance.return_value = Decimal("100000")
        
        mock_sizers = {
            "equity": Mock(spec=PositionSizerProtocol),
            "crypto": Mock(spec=PositionSizerProtocol)
        }
        for sizer in mock_sizers.values():
            sizer.calculate_size.return_value = Decimal("100")
        
        mock_limits = {"equity": [], "crypto": []}
        market_data = {"prices": {"AAPL": 150.0, "BTC-USD": 50000.0}}
        
        # Map symbols
        self.multi_flow.map_symbol_to_classifier("AAPL", "equity")
        self.multi_flow.map_symbol_to_classifier("BTC-USD", "crypto")
        
        # Create and route signals
        signals = [
            Signal(
                signal_id="SIG001",
                strategy_id="momentum",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="SIG002",
                strategy_id="momentum",
                symbol="BTC-USD",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        async def test():
            # Route signals
            for signal in signals:
                await self.multi_flow.route_signal(signal)
            
            # Process all
            orders_by_classifier = await self.multi_flow.process_all_signals(
                mock_portfolios,
                mock_sizers,
                mock_limits,
                market_data
            )
            
            # Should have orders for both classifiers
            self.assertIn("equity", orders_by_classifier)
            self.assertIn("crypto", orders_by_classifier)
            self.assertGreater(len(orders_by_classifier["equity"]), 0)
            self.assertGreater(len(orders_by_classifier["crypto"]), 0)
        
        asyncio.run(test())
    
    def test_missing_classifier_handling(self):
        """Test handling of signals for unmapped symbols."""
        # Create flow manager
        equity_flow = self.multi_flow.create_flow_manager("equity", {})
        equity_flow.register_strategy("momentum", Decimal("1.0"))
        
        # Don't map the symbol
        signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="UNMAPPED",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        async def test():
            # Should handle gracefully
            await self.multi_flow.route_signal(signal)
            
            # Signal should not be in any flow manager
            async with equity_flow._buffer_lock:
                self.assertEqual(len(equity_flow._signal_buffer), 0)
        
        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()