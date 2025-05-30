"""
Integration tests for Risk and Execution modules.

Tests cover:
- Signal to order to fill flow
- Position tracking consistency
- Risk limit enforcement
- Portfolio state updates
- Event flow between modules
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.risk import (
    RiskPortfolioContainer, Signal, SignalType, OrderSide,
    PercentagePositionSizer, MaxPositionLimit, MaxExposureLimit
)
from src.execution import (
    BacktestBrokerRefactored, DefaultExecutionEngine,
    OrderManager, MarketSimulator
)
from src.execution.protocols import OrderStatus, FillType
from src.core.events.types import Event, EventType


class TestRiskExecutionIntegration(unittest.TestCase):
    """Test integration between risk and execution modules."""
    
    def setUp(self):
        """Set up integrated components."""
        # Create risk portfolio
        self.risk_portfolio = RiskPortfolioContainer(
            name="test_portfolio",
            initial_capital=Decimal("100000")
        )
        
        # Set position sizer
        self.risk_portfolio.set_position_sizer(
            PercentagePositionSizer(percentage=Decimal("0.02"))  # 2% per position
        )
        
        # Add risk limits
        self.risk_portfolio.add_risk_limit(
            MaxPositionLimit(max_position_value=Decimal("10000"))
        )
        self.risk_portfolio.add_risk_limit(
            MaxExposureLimit(max_exposure_pct=Decimal("20"))  # 20% max exposure
        )
        
        # Create execution components using risk portfolio's state
        self.broker = BacktestBrokerRefactored(
            portfolio_state=self.risk_portfolio.get_portfolio_state(),
            initial_capital=Decimal("100000")
        )
        
        self.order_manager = OrderManager()
        self.market_simulator = MarketSimulator(
            slippage_model="percentage",
            slippage_params={"percentage": 0.001},
            commission_model="percentage",
            commission_params={"percentage": 0.001}
        )
        
        self.execution_engine = DefaultExecutionEngine(
            broker=self.broker,
            order_manager=self.order_manager,
            market_simulator=self.market_simulator
        )
    
    def test_signal_to_fill_flow(self):
        """Test complete flow from signal generation to fill processing."""
        # Create a buy signal
        signal = Signal(
            signal_id="TEST_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={"reason": "momentum"}
        )
        
        # Market data
        market_data = {
            "timestamp": datetime.now(),
            "prices": {"AAPL": 150.0},
            "AAPL": {
                "bid": 149.95,
                "ask": 150.05,
                "last": 150.00,
                "volume": 10000000
            }
        }
        
        # Process signal through risk module
        orders = self.risk_portfolio.process_signals([signal], market_data)
        
        # Should generate one order
        self.assertEqual(len(orders), 1)
        order = orders[0]
        
        # Check order details
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        
        # Position size should be 2% of capital / price
        expected_size = (Decimal("100000") * Decimal("0.02")) / Decimal("150")
        expected_size = expected_size.quantize(Decimal("1"))  # Round to whole shares
        self.assertEqual(order.quantity, expected_size)
        
        # Now process order through execution engine
        async def run_execution():
            # Add market data to execution engine
            await self.execution_engine.process_event(
                Event(EventType.BAR, market_data["AAPL"], datetime.now())
            )
            
            # Process order
            await self.execution_engine.process_event(
                Event(EventType.ORDER, {"order": order}, datetime.now())
            )
            
            # Check order was filled
            self.assertEqual(len(self.broker._fills), 1)
            fill = self.broker._fills[0]
            
            # Verify fill details
            self.assertEqual(fill.symbol, "AAPL")
            self.assertEqual(fill.quantity, order.quantity)
            self.assertEqual(fill.fill_type, FillType.COMPLETE)
            
            # Verify portfolio state was updated
            portfolio_state = self.risk_portfolio.get_portfolio_state()
            position = portfolio_state.get_position("AAPL")
            
            self.assertIsNotNone(position)
            self.assertEqual(position.quantity, order.quantity)
            self.assertEqual(position.symbol, "AAPL")
            
            # Cash should be reduced
            cash = portfolio_state.get_cash_balance()
            expected_cash = Decimal("100000") - (fill.quantity * fill.price) - fill.commission
            self.assertAlmostEqual(float(cash), float(expected_cash), places=2)
        
        asyncio.run(run_execution())
    
    def test_risk_limit_enforcement(self):
        """Test that risk limits are properly enforced."""
        # Create multiple signals that would exceed exposure limit
        signals = []
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        for i, symbol in enumerate(symbols):
            signal = Signal(
                signal_id=f"TEST_{i:03d}",
                strategy_id="test_strategy",
                symbol=symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            )
            signals.append(signal)
        
        # Market data with high prices
        market_data = {
            "timestamp": datetime.now(),
            "prices": {
                "AAPL": 150.0,
                "GOOGL": 2800.0,
                "MSFT": 380.0,
                "AMZN": 3500.0,
                "TSLA": 1000.0
            }
        }
        
        # Process all signals
        orders = self.risk_portfolio.process_signals(signals, market_data)
        
        # Should not generate orders for all signals due to exposure limit
        # With 20% max exposure and 2% per position, max 10 positions worth
        total_exposure = Decimal("0")
        for order in orders:
            position_value = order.quantity * Decimal(str(market_data["prices"][order.symbol]))
            total_exposure += position_value
        
        # Total exposure should not exceed 20% of capital
        max_allowed = Decimal("100000") * Decimal("0.20")
        self.assertLessEqual(total_exposure, max_allowed)
        
        # Some signals should have been rejected
        self.assertLess(len(orders), len(signals))
    
    def test_position_tracking_consistency(self):
        """Test that position tracking remains consistent across modules."""
        async def run_test():
            # Initial state check
            risk_portfolio_state = self.risk_portfolio.get_portfolio_state()
            broker_portfolio_state = self.broker.portfolio_state
            
            # Should be the same object
            self.assertIs(risk_portfolio_state, broker_portfolio_state)
            
            # Create and process a buy signal
            buy_signal = Signal(
                signal_id="BUY_001",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            )
            
            market_data = {
                "timestamp": datetime.now(),
                "prices": {"AAPL": 150.0},
                "AAPL": {"bid": 149.95, "ask": 150.05, "last": 150.00}
            }
            
            # Process through risk
            orders = self.risk_portfolio.process_signals([buy_signal], market_data)
            self.assertEqual(len(orders), 1)
            
            # Process through execution
            await self.execution_engine.process_event(
                Event(EventType.BAR, market_data["AAPL"], datetime.now())
            )
            await self.execution_engine.process_event(
                Event(EventType.ORDER, {"order": orders[0]}, datetime.now())
            )
            
            # Check position in risk module
            risk_position = risk_portfolio_state.get_position("AAPL")
            self.assertIsNotNone(risk_position)
            
            # Now process a sell signal
            sell_signal = Signal(
                signal_id="SELL_001",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                strength=Decimal("1.0"),
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Update market data
            market_data["prices"]["AAPL"] = 155.0
            market_data["AAPL"]["last"] = 155.0
            
            # Process sell signal
            sell_orders = self.risk_portfolio.process_signals([sell_signal], market_data)
            self.assertEqual(len(sell_orders), 1)
            
            # Process through execution
            await self.execution_engine.process_event(
                Event(EventType.BAR, market_data["AAPL"], datetime.now())
            )
            await self.execution_engine.process_event(
                Event(EventType.ORDER, {"order": sell_orders[0]}, datetime.now())
            )
            
            # Position should be closed
            final_position = risk_portfolio_state.get_position("AAPL")
            self.assertTrue(
                final_position is None or final_position.quantity == Decimal("0")
            )
            
            # Check P&L was recorded
            metrics = risk_portfolio_state.get_risk_metrics()
            self.assertGreater(metrics.realized_pnl, Decimal("0"))  # Made profit
        
        asyncio.run(run_test())
    
    def test_multiple_concurrent_orders(self):
        """Test handling multiple orders concurrently."""
        async def run_test():
            # Create multiple signals
            signals = []
            symbols = ["AAPL", "GOOGL", "MSFT"]
            
            for symbol in symbols:
                signal = Signal(
                    signal_id=f"{symbol}_001",
                    strategy_id="test",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal("0.7"),
                    timestamp=datetime.now(),
                    metadata={}
                )
                signals.append(signal)
            
            # Market data
            market_data = {
                "timestamp": datetime.now(),
                "prices": {
                    "AAPL": 150.0,
                    "GOOGL": 2800.0,
                    "MSFT": 380.0
                }
            }
            
            # Add detailed market data for each symbol
            for symbol in symbols:
                market_data[symbol] = {
                    "bid": market_data["prices"][symbol] - 0.05,
                    "ask": market_data["prices"][symbol] + 0.05,
                    "last": market_data["prices"][symbol]
                }
            
            # Process all signals
            orders = self.risk_portfolio.process_signals(signals, market_data)
            self.assertEqual(len(orders), 3)
            
            # Process all orders through execution
            # First, update market data
            for symbol in symbols:
                await self.execution_engine.process_event(
                    Event(EventType.BAR, market_data[symbol], datetime.now())
                )
            
            # Submit all orders
            tasks = []
            for order in orders:
                task = self.execution_engine.process_event(
                    Event(EventType.ORDER, {"order": order}, datetime.now())
                )
                tasks.append(task)
            
            # Wait for all to complete
            await asyncio.gather(*tasks)
            
            # All should be filled
            self.assertEqual(len(self.broker._fills), 3)
            
            # Check all positions exist
            portfolio_state = self.risk_portfolio.get_portfolio_state()
            for symbol in symbols:
                position = portfolio_state.get_position(symbol)
                self.assertIsNotNone(position)
                self.assertGreater(position.quantity, Decimal("0"))
        
        asyncio.run(run_test())
    
    def test_error_handling_integration(self):
        """Test error handling between modules."""
        # Try to process signal with insufficient capital
        # First, use up most capital
        initial_signal = Signal(
            signal_id="INIT_001",
            strategy_id="test",
            symbol="BRK.A",  # Very expensive stock
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        market_data = {
            "timestamp": datetime.now(),
            "prices": {"BRK.A": 500000.0},  # $500k per share
            "BRK.A": {"bid": 499999, "ask": 500001, "last": 500000}
        }
        
        # This should fail due to position size limits
        orders = self.risk_portfolio.process_signals([initial_signal], market_data)
        
        # Should not generate order due to max position limit ($10k)
        self.assertEqual(len(orders), 0)
        
        # Now try a reasonable order
        reasonable_signal = Signal(
            signal_id="REASONABLE_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.5"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        market_data["prices"]["AAPL"] = 150.0
        market_data["AAPL"] = {"bid": 149.95, "ask": 150.05, "last": 150.0}
        
        orders = self.risk_portfolio.process_signals([reasonable_signal], market_data)
        
        # Should generate order
        self.assertEqual(len(orders), 1)
    
    def test_portfolio_metrics_update(self):
        """Test that portfolio metrics are updated correctly."""
        async def run_test():
            # Get initial metrics
            initial_metrics = self.risk_portfolio.get_portfolio_state().get_risk_metrics()
            self.assertEqual(initial_metrics.total_value, Decimal("100000"))
            self.assertEqual(initial_metrics.positions_value, Decimal("0"))
            
            # Process a buy order
            signal = Signal(
                signal_id="TEST_001",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            )
            
            market_data = {
                "timestamp": datetime.now(),
                "prices": {"AAPL": 150.0},
                "AAPL": {"bid": 149.95, "ask": 150.05, "last": 150.0}
            }
            
            orders = self.risk_portfolio.process_signals([signal], market_data)
            
            # Execute order
            await self.execution_engine.process_event(
                Event(EventType.BAR, market_data["AAPL"], datetime.now())
            )
            await self.execution_engine.process_event(
                Event(EventType.ORDER, {"order": orders[0]}, datetime.now())
            )
            
            # Update market prices
            self.risk_portfolio.update_market_data(market_data)
            
            # Get updated metrics
            updated_metrics = self.risk_portfolio.get_portfolio_state().get_risk_metrics()
            
            # Total value should still be close to initial (minus commission/slippage)
            self.assertAlmostEqual(
                float(updated_metrics.total_value),
                float(initial_metrics.total_value),
                delta=100  # Allow for commission/slippage
            )
            
            # Should have positions value > 0
            self.assertGreater(updated_metrics.positions_value, Decimal("0"))
            
            # Cash should be reduced
            self.assertLess(updated_metrics.cash_balance, initial_metrics.cash_balance)
        
        asyncio.run(run_test())


class TestEventFlowIntegration(unittest.TestCase):
    """Test event flow between modules."""
    
    def setUp(self):
        """Set up components with event tracking."""
        self.events_published = []
        
        # Mock container with event bus
        self.mock_container = Mock()
        self.mock_container.publish_event = Mock(side_effect=self._track_event)
        
        # Create risk portfolio with mock parent
        self.risk_portfolio = RiskPortfolioContainer(
            name="test_portfolio",
            initial_capital=Decimal("100000")
        )
        self.risk_portfolio.parent = self.mock_container
    
    def _track_event(self, event_type, event_data):
        """Track published events."""
        self.events_published.append((event_type, event_data))
    
    def test_order_event_flow(self):
        """Test events are properly published during order flow."""
        signal = Signal(
            signal_id="TEST_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        market_data = {
            "timestamp": datetime.now(),
            "prices": {"AAPL": 150.0}
        }
        
        # Process signal
        orders = self.risk_portfolio.process_signals([signal], market_data)
        
        # Should have published order created event
        order_events = [
            e for e in self.events_published
            if e[1].payload.get("type") == "order_created"
        ]
        self.assertGreater(len(order_events), 0)
        
        # Check event details
        event_type, event_data = order_events[0]
        self.assertEqual(event_type, EventType.ORDER)
        self.assertEqual(event_data.source_id, "test_portfolio")
        self.assertIn("order", event_data.payload)
        self.assertIn("signal", event_data.payload)


if __name__ == "__main__":
    unittest.main()