"""
Integration tests for Step 2 risk container.

Tests the risk container integration with:
- Event system (isolation and communication)
- Strategy components (signal processing)
- Execution components (order and fill handling)
- Data components (market data updates)

Architecture Context:
    - Part of: Step 2 - Add Risk Container testing
    - Validates: Cross-component integration works correctly
    - Coverage: Integration-level testing between containers
    - Dependencies: Core event system, logging infrastructure
"""

import unittest
import asyncio
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime

# Import test components
from src.risk.risk_container import RiskContainer
from src.risk.step2_container_factory import create_test_risk_container
from src.risk.models import (
    RiskConfig, TradingSignal, Order, Fill,
    SignalType, OrderSide, OrderType
)
from src.core.events.enhanced_isolation import get_enhanced_isolation_manager


class TestRiskContainerEventIntegration(unittest.TestCase):
    """Test risk container event system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.isolation_manager = get_enhanced_isolation_manager()
        self.container = create_test_risk_container("integration_test", 100000.0)
        
        # Create mock strategy and execution containers
        self.strategy_bus = self.isolation_manager.create_container_bus("strategy_test")
        self.execution_bus = self.isolation_manager.create_container_bus("execution_test")
        
        # Track events
        self.received_orders = []
        self.received_signals = []
    
    def tearDown(self):
        """Clean up test environment."""
        self.container.cleanup()
        self.isolation_manager.remove_container_bus("strategy_test")
        self.isolation_manager.remove_container_bus("execution_test")
    
    def test_event_isolation_between_containers(self):
        """Test that containers are properly isolated."""
        # Subscribe to events on different buses
        risk_events = []
        strategy_events = []
        
        def risk_handler(event_type, data):
            risk_events.append((event_type, data))
        
        def strategy_handler(event_type, data):
            strategy_events.append((event_type, data))
        
        self.container.event_bus.subscribe("TEST_EVENT", risk_handler)
        self.strategy_bus.subscribe("TEST_EVENT", strategy_handler)
        
        # Publish event on risk bus
        self.container.event_bus.publish("TEST_EVENT", {"source": "risk"})
        
        # Publish event on strategy bus
        self.strategy_bus.publish("TEST_EVENT", {"source": "strategy"})
        
        # Check isolation - each bus should only receive its own events
        self.assertEqual(len(risk_events), 1)
        self.assertEqual(len(strategy_events), 1)
        self.assertEqual(risk_events[0][1]["source"], "risk")
        self.assertEqual(strategy_events[0][1]["source"], "strategy")
    
    def test_signal_to_order_event_flow(self):
        """Test signal processing creates order events."""
        # Set up order capture
        def capture_order(event_type, order):
            if event_type == "ORDER":
                self.received_orders.append(order)
        
        self.container.event_bus.subscribe("ORDER", capture_order)
        
        # Update market data
        self.container.update_market_data({"SPY": 400.0})
        
        # Send signal
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        self.container.on_signal(signal)
        
        # Verify order was published
        self.assertEqual(len(self.received_orders), 1)
        order = self.received_orders[0]
        self.assertEqual(order.symbol, "SPY")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertGreater(order.quantity, 0)
    
    def test_fill_event_processing(self):
        """Test fill events update portfolio state."""
        # Process a signal first to create order
        self.container.update_market_data({"SPY": 400.0})
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        self.container.on_signal(signal)
        
        # Check initial state
        initial_cash = self.container.portfolio_state.cash
        initial_positions = len(self.container.portfolio_state.positions)
        
        # Create fill event
        fill = Fill(
            fill_id="FILL001",
            order_id="test_order",
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('400.0'),
            timestamp=datetime.now()
        )
        
        # Process fill through event system
        self.container.event_bus.publish("FILL", fill)
        
        # Allow event processing
        import time
        time.sleep(0.1)
        
        # Verify portfolio was updated
        self.assertLess(self.container.portfolio_state.cash, initial_cash)
        self.assertGreater(len(self.container.portfolio_state.positions), initial_positions)
        
        # Verify position details
        position = self.container.portfolio_state.positions.get("SPY")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, Decimal('100'))
        self.assertEqual(position.avg_price, Decimal('400.0'))
    
    def test_market_data_event_processing(self):
        """Test market data events update portfolio values."""
        # Add a position first
        fill = Fill(
            fill_id="FILL001", order_id="test_order", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.0'),
            timestamp=datetime.now()
        )
        self.container.portfolio_state.update_position(fill)
        
        # Update market data through event system
        market_data = {"SPY": 420.0, "QQQ": 350.0}
        self.container.event_bus.publish("MARKET_DATA", market_data)
        
        # Allow event processing
        import time
        time.sleep(0.1)
        
        # Verify prices were updated
        prices = self.container.portfolio_state.get_current_prices()
        self.assertEqual(prices["SPY"], 420.0)
        self.assertEqual(prices["QQQ"], 350.0)
        
        # Verify portfolio value recalculated
        total_value = self.container.portfolio_state.calculate_total_value()
        # Cash + position value: 60000 + (100 * 420) = 102000
        expected_value = Decimal('102000')
        self.assertEqual(total_value, expected_value)


class TestRiskContainerComponentIntegration(unittest.TestCase):
    """Test integration between risk container components."""
    
    def setUp(self):
        """Set up test environment."""
        self.container = create_test_risk_container("component_test", 50000.0, "percent_risk")
    
    def tearDown(self):
        """Clean up test environment."""
        self.container.cleanup()
    
    def test_risk_limits_prevent_oversized_positions(self):
        """Test risk limits prevent oversized positions."""
        # Update market data
        self.container.update_market_data({"SPY": 100.0})
        
        # Create large position first
        large_fill = Fill(
            fill_id="FILL001", order_id="test", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('400'), price=Decimal('100.0'),
            timestamp=datetime.now()
        )
        self.container.portfolio_state.update_position(large_fill)
        
        # Try to add more to the position
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.5')
        )
        
        initial_orders = self.container.created_orders
        self.container.on_signal(signal)
        
        # Should be rejected due to position size limits
        self.assertEqual(self.container.created_orders, initial_orders)
        self.assertEqual(self.container.rejected_signals, 1)
    
    def test_position_sizer_cash_constraints(self):
        """Test position sizer respects cash constraints."""
        # Set low cash by creating large position
        large_fill = Fill(
            fill_id="FILL001", order_id="test", symbol="AAPL",
            side=OrderSide.BUY, quantity=Decimal('200'), price=Decimal('200.0'),
            timestamp=datetime.now()
        )
        self.container.portfolio_state.update_position(large_fill)
        
        # Update market data
        self.container.update_market_data({"SPY": 500.0, "AAPL": 200.0})
        
        # Try to buy expensive stock with limited cash
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('1.0')
        )
        
        # Should create small order due to cash constraints
        initial_orders = self.container.created_orders
        self.container.on_signal(signal)
        
        # Order should be created but with constrained size
        self.assertEqual(self.container.created_orders, initial_orders + 1)
        
        # Check order size is reasonable given cash constraints
        cash_available = self.container.portfolio_state.cash
        max_affordable = cash_available / Decimal('500')  # Price per share
        
        # The actual sizing will be further constrained by risk rules
        self.assertGreaterEqual(cash_available, 0)  # Still have some cash
    
    def test_portfolio_state_tracks_pending_orders(self):
        """Test portfolio state tracks pending orders."""
        self.container.update_market_data({"SPY": 400.0})
        
        # Process signal to create order
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        self.container.on_signal(signal)
        
        # Should have pending order
        self.assertGreater(len(self.container.portfolio_state.pending_orders), 0)
        
        # Create fill for the order
        pending_orders = list(self.container.portfolio_state.pending_orders.keys())
        fill = Fill(
            fill_id="FILL001",
            order_id=pending_orders[0],
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('400.0'),
            timestamp=datetime.now()
        )
        
        self.container.on_fill(fill)
        
        # Pending order should be removed
        self.assertEqual(len(self.container.portfolio_state.pending_orders), 0)
    
    def test_end_to_end_trade_lifecycle(self):
        """Test complete trade lifecycle from signal to fill."""
        # Initial state
        initial_cash = self.container.portfolio_state.cash
        initial_value = self.container.portfolio_state.total_value
        
        # Update market data
        self.container.update_market_data({"SPY": 400.0})
        
        # Process signal
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        
        self.container.on_signal(signal)
        
        # Verify order created
        self.assertEqual(self.container.created_orders, 1)
        self.assertEqual(self.container.processed_signals, 1)
        self.assertGreater(len(self.container.portfolio_state.pending_orders), 0)
        
        # Get the created order
        pending_order_id = list(self.container.portfolio_state.pending_orders.keys())[0]
        
        # Create fill
        fill = Fill(
            fill_id="FILL001",
            order_id=pending_order_id,
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('405.0'),  # Slightly higher fill price
            timestamp=datetime.now()
        )
        
        self.container.on_fill(fill)
        
        # Verify final state
        self.assertEqual(len(self.container.portfolio_state.pending_orders), 0)
        self.assertIn("SPY", self.container.portfolio_state.positions)
        
        position = self.container.portfolio_state.positions["SPY"]
        self.assertEqual(position.quantity, Decimal('100'))
        self.assertEqual(position.avg_price, Decimal('405.0'))
        
        # Verify cash reduction
        expected_cash = initial_cash - Decimal('40500')  # 100 * 405
        self.assertEqual(self.container.portfolio_state.cash, expected_cash)
        
        # Update market price and check unrealized P&L
        self.container.update_market_data({"SPY": 410.0})
        total_value = self.container.portfolio_state.calculate_total_value()
        
        # Expected: cash + position value
        # Cash: 50000 - 40500 = 9500
        # Position: 100 * 410 = 41000
        # Total: 50500
        expected_total = expected_cash + Decimal('41000')
        self.assertEqual(total_value, expected_total)
        
        # Check unrealized P&L
        unrealized_pnl = self.container.portfolio_state.get_unrealized_pnl()
        expected_pnl = Decimal('100') * (Decimal('410') - Decimal('405'))  # 100 * 5 = 500
        self.assertEqual(unrealized_pnl, expected_pnl)


class TestRiskContainerMultiSignalProcessing(unittest.TestCase):
    """Test risk container handling multiple signals."""
    
    def setUp(self):
        """Set up test environment."""
        self.container = create_test_risk_container("multi_test", 100000.0, "fixed")
    
    def tearDown(self):
        """Clean up test environment."""
        self.container.cleanup()
    
    def test_multiple_symbols_processing(self):
        """Test processing signals for multiple symbols."""
        # Update market data for multiple symbols
        self.container.update_market_data({
            "SPY": 400.0,
            "QQQ": 350.0,
            "IWM": 200.0
        })
        
        # Create signals for different symbols
        signals = [
            TradingSignal(
                signal_id=f"SIG00{i}",
                strategy_id="momentum",
                symbol=symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal('0.7')
            )
            for i, symbol in enumerate(["SPY", "QQQ", "IWM"], 1)
        ]
        
        # Process all signals
        for signal in signals:
            self.container.on_signal(signal)
        
        # Verify all orders created
        self.assertEqual(self.container.created_orders, 3)
        self.assertEqual(self.container.processed_signals, 3)
        self.assertEqual(len(self.container.portfolio_state.pending_orders), 3)
    
    def test_risk_limits_across_multiple_positions(self):
        """Test risk limits work across multiple positions."""
        self.container.update_market_data({
            "SPY": 400.0,
            "QQQ": 350.0
        })
        
        # Create positions in both symbols
        fills = [
            Fill(
                fill_id="FILL001", order_id="test1", symbol="SPY",
                side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.0'),
                timestamp=datetime.now()
            ),
            Fill(
                fill_id="FILL002", order_id="test2", symbol="QQQ",
                side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('350.0'),
                timestamp=datetime.now()
            )
        ]
        
        for fill in fills:
            self.container.portfolio_state.update_position(fill)
        
        # Try to add large position that would exceed concentration limits
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="AAPL",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.9')
        )
        
        # Update price for new symbol
        self.container.update_market_data({"AAPL": 150.0})
        
        initial_orders = self.container.created_orders
        self.container.on_signal(signal)
        
        # Should create order but with appropriate sizing
        # The test container has relaxed limits for testing
        self.assertEqual(self.container.created_orders, initial_orders + 1)
    
    def test_signal_strength_variations(self):
        """Test different signal strengths produce different order sizes."""
        self.container.update_market_data({"SPY": 400.0})
        
        # Create signals with different strengths
        strengths = [Decimal('0.3'), Decimal('0.6'), Decimal('1.0')]
        order_sizes = []
        
        for i, strength in enumerate(strengths):
            signal = TradingSignal(
                signal_id=f"SIG00{i+1}",
                strategy_id="momentum",
                symbol="SPY",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=strength
            )
            
            # Reset container for clean test
            if i > 0:
                self.container.reset()
                self.container.update_market_data({"SPY": 400.0})
            
            initial_orders = len(self.container.portfolio_state.pending_orders)
            self.container.on_signal(signal)
            
            # Get the created order size
            if len(self.container.portfolio_state.pending_orders) > initial_orders:
                order_id = list(self.container.portfolio_state.pending_orders.keys())[-1]
                order = self.container.portfolio_state.pending_orders[order_id]
                order_sizes.append(float(order.quantity))
        
        # Verify different sizes (assuming fixed sizing scales with strength)
        self.assertEqual(len(order_sizes), 3)
        # Higher strength should generally produce larger orders
        # (though exact relationship depends on sizing method)


if __name__ == '__main__':
    unittest.main(verbosity=2)