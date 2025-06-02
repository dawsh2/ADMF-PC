"""
System tests for Step 2: Complete pipeline with risk container.

Tests the full trading pipeline:
Strategy → Risk → Execution flow with Step 2 risk container

Validates:
- Complete signal-to-fill workflow
- Risk management integration
- Event isolation between containers
- Portfolio state consistency
- Performance and error handling

Architecture Context:
    - Part of: Step 2 - Add Risk Container system validation
    - Validates: End-to-end system behavior with risk management
    - Coverage: System-level testing of complete trading pipeline
    - Dependencies: All system components (strategy, risk, execution, data)
"""

import unittest
import time
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime
from typing import List, Dict, Any

# Import system components
from src.risk.risk_container import RiskContainer
from src.risk.step2_container_factory import create_test_risk_container
from src.risk.models import (
    RiskConfig, TradingSignal, Order, Fill,
    SignalType, OrderSide, OrderType
)

# Import strategy components for system test
from src.strategy.strategies.momentum import MomentumStrategy
from src.strategy.components.indicators import SimpleMovingAverage

# Import core infrastructure
from src.core.events.enhanced_isolation import get_enhanced_isolation_manager
from src.core.logging.structured import ContainerLogger
from src.data.models import MarketData


class MockExecutionEngine:
    """Mock execution engine for system testing."""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(f"{container_id}_execution")
        self.logger = ContainerLogger("MockExecution", container_id, "execution")
        
        # Track orders and fills
        self.received_orders: List[Order] = []
        self.generated_fills: List[Fill] = []
        self.fill_delay = 0.1  # Simulate execution delay
        
        # Market simulation
        self.current_prices = {}
        self.slippage = 0.001  # 0.1% slippage
        
        # Subscribe to orders
        self.event_bus.subscribe("ORDER", self.on_order)
        
        self.logger.info("MockExecutionEngine initialized")
    
    def on_order(self, event_type: str, order: Order) -> None:
        """Process incoming order."""
        self.received_orders.append(order)
        
        self.logger.info(
            "Order received",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=float(order.quantity)
        )
        
        # Simulate execution after delay
        self._execute_order(order)
    
    def _execute_order(self, order: Order) -> None:
        """Simulate order execution."""
        # Get current price
        current_price = self.current_prices.get(order.symbol)
        if not current_price:
            self.logger.warning(f"No price for {order.symbol}, cannot execute")
            return
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage)
        else:
            fill_price = current_price * (1 - self.slippage)
        
        # Create fill
        fill = Fill(
            fill_id=f"FILL_{len(self.generated_fills)+1:04d}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=Decimal(str(fill_price)),
            timestamp=datetime.now()
        )
        
        self.generated_fills.append(fill)
        
        # Publish fill event
        self.event_bus.publish("FILL", fill)
        
        self.logger.info(
            "Order executed",
            fill_id=fill.fill_id,
            order_id=order.order_id,
            price=float(fill.price),
            slippage=self.slippage
        )
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update market prices."""
        self.current_prices.update(prices)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.isolation_manager.remove_container_bus(f"{self.container_id}_execution")


class MockStrategyEngine:
    """Mock strategy engine for system testing."""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(f"{container_id}_strategy")
        self.logger = ContainerLogger("MockStrategy", container_id, "strategy")
        
        # Strategy state
        self.generated_signals: List[TradingSignal] = []
        self.signal_counter = 0
        
        # Simple momentum strategy components
        self.sma_short = SimpleMovingAverage(5)
        self.sma_long = SimpleMovingAverage(20)
        self.price_history = {}
        
        self.logger.info("MockStrategyEngine initialized")
    
    def update_market_data(self, market_data: Dict[str, float]) -> None:
        """Process market data and generate signals."""
        for symbol, price in market_data.items():
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(price)
            
            # Keep only recent prices
            if len(self.price_history[symbol]) > 25:
                self.price_history[symbol] = self.price_history[symbol][-25:]
            
            # Generate signal if we have enough data
            if len(self.price_history[symbol]) >= 20:
                signal = self._check_momentum_signal(symbol, price)
                if signal:
                    self.generated_signals.append(signal)
                    self.event_bus.publish("SIGNAL", signal)
                    
                    self.logger.info(
                        "Signal generated",
                        signal_id=signal.signal_id,
                        symbol=symbol,
                        side=signal.side.value,
                        strength=float(signal.strength)
                    )
    
    def _check_momentum_signal(self, symbol: str, current_price: float) -> TradingSignal:
        """Check for momentum signals."""
        prices = self.price_history[symbol]
        
        # Update moving averages
        for price in prices[-5:]:  # Update with recent prices
            self.sma_short.update(price)
        
        for price in prices:  # Update long MA with all prices
            self.sma_long.update(price)
        
        # Generate signal based on MA crossover
        if not self.sma_short.is_ready() or not self.sma_long.is_ready():
            return None
        
        short_ma = self.sma_short.get_value()
        long_ma = self.sma_long.get_value()
        
        # Simple momentum signal
        if short_ma > long_ma * 1.01:  # 1% threshold
            # Bullish signal
            self.signal_counter += 1
            strength = min(abs(short_ma - long_ma) / long_ma * 10, 1.0)  # Normalize strength
            
            return TradingSignal(
                signal_id=f"SIG_{self.signal_counter:04d}",
                strategy_id="momentum",
                symbol=symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal(str(strength)),
                metadata={
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "price": current_price
                }
            )
        
        return None
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.isolation_manager.remove_container_bus(f"{self.container_id}_strategy")


class TestStep2SystemValidation(unittest.TestCase):
    """System validation tests for Step 2."""
    
    def setUp(self):
        """Set up system test environment."""
        self.isolation_manager = get_enhanced_isolation_manager()
        
        # Create system components
        self.risk_container = create_test_risk_container("system_test", 100000.0, "percent_risk")
        self.strategy_engine = MockStrategyEngine("system_test")
        self.execution_engine = MockExecutionEngine("system_test")
        
        # Wire components together
        self._wire_components()
        
        # Test state tracking
        self.processed_signals = []
        self.created_orders = []
        self.executed_fills = []
        
        self.logger = ContainerLogger("SystemTest", "system_test", "system_test")
        
    def tearDown(self):
        """Clean up system test environment."""
        self.risk_container.cleanup()
        self.strategy_engine.cleanup()
        self.execution_engine.cleanup()
    
    def _wire_components(self) -> None:
        """Wire components together with cross-container event subscriptions."""
        # Strategy → Risk: Signal flow
        def on_strategy_signal(event_type: str, signal: TradingSignal):
            if event_type == "SIGNAL":
                self.processed_signals.append(signal)
                self.risk_container.on_signal(signal)
        
        self.strategy_engine.event_bus.subscribe("SIGNAL", on_strategy_signal)
        
        # Risk → Execution: Order flow
        def on_risk_order(event_type: str, order: Order):
            if event_type == "ORDER":
                self.created_orders.append(order)
                self.execution_engine.event_bus.publish("ORDER", order)
        
        self.risk_container.event_bus.subscribe("ORDER", on_risk_order)
        
        # Execution → Risk: Fill flow
        def on_execution_fill(event_type: str, fill: Fill):
            if event_type == "FILL":
                self.executed_fills.append(fill)
                self.risk_container.on_fill(fill)
        
        self.execution_engine.event_bus.subscribe("FILL", on_execution_fill)
    
    def test_complete_signal_to_fill_pipeline(self):
        """Test complete trading pipeline from signal generation to fill processing."""
        # Set up market data
        market_data = {"SPY": 400.0}
        
        # Update all components with market data
        self.risk_container.update_market_data(market_data)
        self.execution_engine.update_prices(market_data)
        
        # Simulate price movement to generate signals
        price_sequence = [400.0, 401.0, 402.0, 403.0, 405.0, 407.0, 410.0]
        
        for price in price_sequence:
            market_data = {"SPY": price}
            
            # Update all components
            self.risk_container.update_market_data(market_data)
            self.execution_engine.update_prices(market_data)
            self.strategy_engine.update_market_data(market_data)
            
            # Allow event processing
            time.sleep(0.01)
        
        # Verify pipeline worked
        self.assertGreater(len(self.processed_signals), 0, "No signals generated")
        self.assertGreater(len(self.created_orders), 0, "No orders created")
        self.assertGreater(len(self.executed_fills), 0, "No fills executed")
        
        # Verify data consistency
        self.assertEqual(len(self.created_orders), len(self.executed_fills), 
                        "Orders and fills count mismatch")
        
        # Verify portfolio state
        portfolio_state = self.risk_container.get_state()
        self.assertGreater(portfolio_state['portfolio_value'], 0)
        
        if portfolio_state['positions_count'] > 0:
            self.assertIn("SPY", self.risk_container.portfolio_state.positions)
            position = self.risk_container.portfolio_state.positions["SPY"]
            self.assertGreater(position.quantity, 0)
    
    def test_risk_management_in_pipeline(self):
        """Test risk management prevents oversized positions."""
        # Set up market data
        market_data = {"SPY": 100.0}  # Low price for easier position size testing
        
        self.risk_container.update_market_data(market_data)
        self.execution_engine.update_prices(market_data)
        
        # Create multiple large signals
        large_signals = []
        for i in range(5):
            signal = TradingSignal(
                signal_id=f"LARGE_SIG_{i:03d}",
                strategy_id="test",
                symbol="SPY",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal('1.0')  # Maximum strength
            )
            large_signals.append(signal)
        
        # Process signals
        initial_orders = len(self.created_orders)
        initial_rejected = self.risk_container.rejected_signals
        
        for signal in large_signals:
            self.risk_container.on_signal(signal)
            time.sleep(0.01)  # Allow processing
        
        # Some signals should be rejected due to risk limits
        total_orders = len(self.created_orders) - initial_orders
        total_rejected = self.risk_container.rejected_signals - initial_rejected
        
        self.assertGreater(total_rejected, 0, "Risk limits did not reject any signals")
        self.assertLess(total_orders, len(large_signals), "All signals were accepted")
        
        # Verify portfolio didn't exceed risk limits
        portfolio_state = self.risk_container.get_state()
        if portfolio_state['positions_count'] > 0:
            # Check position size relative to portfolio
            exposure = self.risk_container.portfolio_state.get_exposure()
            gross_exposure_pct = exposure['gross_exposure'] / portfolio_state['portfolio_value']
            self.assertLessEqual(gross_exposure_pct, 0.5, "Position exceeded concentration limits")
    
    def test_event_isolation_in_system(self):
        """Test event isolation works in complete system."""
        # Create event trackers for each component
        risk_events = []
        strategy_events = []
        execution_events = []
        
        def track_risk_events(event_type, data):
            risk_events.append((event_type, data))
        
        def track_strategy_events(event_type, data):
            strategy_events.append((event_type, data))
        
        def track_execution_events(event_type, data):
            execution_events.append((event_type, data))
        
        # Subscribe to test events on each bus
        self.risk_container.event_bus.subscribe("TEST_EVENT", track_risk_events)
        self.strategy_engine.event_bus.subscribe("TEST_EVENT", track_strategy_events)
        self.execution_engine.event_bus.subscribe("TEST_EVENT", track_execution_events)
        
        # Publish test events on each bus
        self.risk_container.event_bus.publish("TEST_EVENT", {"source": "risk"})
        self.strategy_engine.event_bus.publish("TEST_EVENT", {"source": "strategy"})
        self.execution_engine.event_bus.publish("TEST_EVENT", {"source": "execution"})
        
        # Verify isolation
        self.assertEqual(len(risk_events), 1)
        self.assertEqual(len(strategy_events), 1)
        self.assertEqual(len(execution_events), 1)
        
        self.assertEqual(risk_events[0][1]["source"], "risk")
        self.assertEqual(strategy_events[0][1]["source"], "strategy")
        self.assertEqual(execution_events[0][1]["source"], "execution")
    
    def test_portfolio_consistency_under_load(self):
        """Test portfolio state consistency under multiple concurrent operations."""
        # Set up multiple symbols
        symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        market_data = {symbol: 100.0 + i * 10 for i, symbol in enumerate(symbols)}
        
        self.risk_container.update_market_data(market_data)
        self.execution_engine.update_prices(market_data)
        
        # Generate multiple signals for different symbols
        signals = []
        for i, symbol in enumerate(symbols):
            for strength in [0.3, 0.6, 0.9]:
                signal = TradingSignal(
                    signal_id=f"LOAD_SIG_{i}_{int(strength*10)}",
                    strategy_id="load_test",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal(str(strength))
                )
                signals.append(signal)
        
        # Process signals rapidly
        for signal in signals:
            self.risk_container.on_signal(signal)
            time.sleep(0.001)  # Minimal delay
        
        # Allow all processing to complete
        time.sleep(0.1)
        
        # Verify portfolio consistency
        portfolio_state = self.risk_container.portfolio_state
        
        # Calculate total value manually
        manual_total = portfolio_state.cash
        for symbol, position in portfolio_state.positions.items():
            current_price = portfolio_state.current_prices.get(symbol, 0)
            manual_total += position.quantity * Decimal(str(current_price))
        
        calculated_total = portfolio_state.calculate_total_value()
        
        self.assertEqual(manual_total, calculated_total, 
                        "Portfolio value calculation inconsistency")
        
        # Verify no negative cash (basic sanity check)
        self.assertGreaterEqual(float(portfolio_state.cash), 0, 
                               "Portfolio has negative cash")
    
    def test_error_handling_in_pipeline(self):
        """Test system handles errors gracefully."""
        # Test with invalid signal
        invalid_signal = TradingSignal(
            signal_id="INVALID",
            strategy_id="test",
            symbol="INVALID_SYMBOL",  # No price data
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        # Should not crash the system
        initial_orders = self.risk_container.created_orders
        initial_rejected = self.risk_container.rejected_signals
        
        self.risk_container.on_signal(invalid_signal)
        
        # Should either reject or handle gracefully
        final_orders = self.risk_container.created_orders
        final_rejected = self.risk_container.rejected_signals
        
        # Either rejected or no order created
        self.assertTrue(final_rejected > initial_rejected or final_orders == initial_orders)
        
        # System should still be functional
        valid_signal = TradingSignal(
            signal_id="VALID",
            strategy_id="test",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        self.risk_container.update_market_data({"SPY": 400.0})
        self.risk_container.on_signal(valid_signal)
        
        # Should process valid signal
        self.assertGreater(self.risk_container.created_orders, initial_orders)
    
    def test_performance_characteristics(self):
        """Test system performance under typical load."""
        # Set up market data
        market_data = {"SPY": 400.0}
        self.risk_container.update_market_data(market_data)
        self.execution_engine.update_prices(market_data)
        
        # Generate batch of signals
        signals = []
        for i in range(100):
            signal = TradingSignal(
                signal_id=f"PERF_SIG_{i:03d}",
                strategy_id="performance_test",
                symbol="SPY",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal(str(0.1 + (i % 9) * 0.1))  # Vary strength
            )
            signals.append(signal)
        
        # Time signal processing
        start_time = time.time()
        
        for signal in signals:
            self.risk_container.on_signal(signal)
        
        processing_time = time.time() - start_time
        
        # Performance assertions (adjust based on requirements)
        avg_time_per_signal = processing_time / len(signals)
        self.assertLess(avg_time_per_signal, 0.01,  # Less than 10ms per signal
                       f"Signal processing too slow: {avg_time_per_signal:.4f}s per signal")
        
        # Verify processing completed successfully
        self.assertGreater(self.risk_container.processed_signals, 0)
        
        # Check memory usage hasn't grown excessively
        state = self.risk_container.get_state()
        self.assertLessEqual(state['pending_orders'], 50,  # Reasonable pending orders limit
                            "Too many pending orders accumulated")


class TestStep2RequirementsValidation(unittest.TestCase):
    """Validate all Step 2 requirements are met."""
    
    def setUp(self):
        """Set up validation environment."""
        self.container = create_test_risk_container("validation", 100000.0)
    
    def tearDown(self):
        """Clean up validation environment."""
        self.container.cleanup()
    
    def test_requirement_risk_container_isolation(self):
        """Validate: Risk container has proper event isolation."""
        # Requirement: Risk container uses isolated event bus
        self.assertIsNotNone(self.container.event_bus)
        self.assertTrue(self.container.event_bus.container_id.endswith("_risk"))
        
        # Test isolation
        external_events = []
        def external_handler(event_type, data):
            external_events.append((event_type, data))
        
        # Create external bus
        isolation_manager = get_enhanced_isolation_manager()
        external_bus = isolation_manager.create_container_bus("external_test")
        external_bus.subscribe("TEST", external_handler)
        
        # Publish on risk container bus
        self.container.event_bus.publish("TEST", {"data": "risk"})
        
        # External bus should not receive events
        self.assertEqual(len(external_events), 0)
        
        isolation_manager.remove_container_bus("external_test")
    
    def test_requirement_position_tracking(self):
        """Validate: Risk container tracks positions correctly."""
        # Requirement: Portfolio state tracks positions and cash
        initial_cash = self.container.portfolio_state.cash
        
        # Create fill
        fill = Fill(
            fill_id="VAL_FILL", order_id="VAL_ORDER", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400'),
            timestamp=datetime.now()
        )
        
        self.container.on_fill(fill)
        
        # Verify position created
        self.assertIn("SPY", self.container.portfolio_state.positions)
        position = self.container.portfolio_state.positions["SPY"]
        self.assertEqual(position.quantity, Decimal('100'))
        
        # Verify cash updated
        expected_cash = initial_cash - Decimal('40000')
        self.assertEqual(self.container.portfolio_state.cash, expected_cash)
    
    def test_requirement_risk_limits(self):
        """Validate: Risk limits enforce constraints."""
        # Requirement: Risk limits prevent oversized positions
        self.container.update_market_data({"SPY": 100.0})
        
        # Create large signal that should be rejected
        large_signal = TradingSignal(
            signal_id="LARGE", strategy_id="test", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, 
            strength=Decimal('1.0')  # Very high strength
        )
        
        # Process multiple large signals
        initial_rejected = self.container.rejected_signals
        for _ in range(10):
            self.container.on_signal(large_signal)
        
        # Should have some rejections
        self.assertGreater(self.container.rejected_signals, initial_rejected)
    
    def test_requirement_position_sizing(self):
        """Validate: Position sizing works correctly."""
        # Requirement: Position sizer calculates appropriate sizes
        self.container.update_market_data({"SPY": 400.0})
        
        signal = TradingSignal(
            signal_id="SIZE_TEST", strategy_id="test", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.5')
        )
        
        initial_orders = self.container.created_orders
        self.container.on_signal(signal)
        
        # Should create order
        self.assertEqual(self.container.created_orders, initial_orders + 1)
        
        # Order size should be reasonable
        pending_orders = self.container.portfolio_state.pending_orders
        if pending_orders:
            order = list(pending_orders.values())[0]
            self.assertGreater(order.quantity, 0)
            
            # Size should be constrained
            order_value = float(order.quantity) * 400.0
            portfolio_value = float(self.container.portfolio_state.total_value)
            position_pct = order_value / portfolio_value
            self.assertLessEqual(position_pct, 0.5)  # Should not exceed 50% (test config)
    
    def test_requirement_order_management(self):
        """Validate: Order management creates proper orders."""
        # Requirement: Order manager creates valid orders with metadata
        self.container.update_market_data({"SPY": 400.0})
        
        signal = TradingSignal(
            signal_id="ORDER_TEST", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.7'),
            metadata={"test_meta": "test_value"}
        )
        
        self.container.on_signal(signal)
        
        # Check order was created
        pending_orders = self.container.portfolio_state.pending_orders
        self.assertGreater(len(pending_orders), 0)
        
        # Validate order properties
        order = list(pending_orders.values())[0]
        self.assertEqual(order.symbol, "SPY")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertIn("test_meta", order.metadata)
        self.assertEqual(order.metadata["test_meta"], "test_value")
    
    def test_requirement_event_flow_logging(self):
        """Validate: System logs event flows properly."""
        # Requirement: All major events are logged with structured data
        # This is validated by checking that logging calls don't raise exceptions
        # and that components have proper logger setup
        
        self.assertIsNotNone(self.container.logger)
        self.assertEqual(self.container.logger.component_name, "RiskContainer")
        
        # Test logging during normal operation
        self.container.update_market_data({"SPY": 400.0})
        
        signal = TradingSignal(
            signal_id="LOG_TEST", strategy_id="test", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        
        # Should not raise exceptions
        try:
            self.container.on_signal(signal)
        except Exception as e:
            self.fail(f"Signal processing raised exception: {e}")


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)