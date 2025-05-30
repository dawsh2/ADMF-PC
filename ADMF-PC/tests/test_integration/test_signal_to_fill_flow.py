"""
Integration test for complete signal to fill flow.

This test validates the entire flow:
1. Strategy generates signals
2. Signals flow through risk management
3. Orders are created and sent to execution
4. Fills are generated and sent back to risk
5. Portfolio state is updated correctly
"""

import unittest
import asyncio
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
    OrderType
)
from src.execution import OrderStatus, FillStatus
from src.risk.risk_portfolio import RiskPortfolioContainer
from src.risk.risk_limits import MaxPositionLimit, MaxExposureLimit
from src.risk.signal_flow import SignalFlowManager
from src.execution.execution_engine import DefaultExecutionEngine
from src.execution.brokers.backtest_broker import BacktestBrokerRefactored
from src.core.events.event_bus import EventBus
from src.core.events.types import EventType, Event
from tests.test_strategies.test_example_strategies import SimpleMomentumStrategy


class TestSignalToFillFlow(unittest.TestCase):
    """Test complete signal to fill flow."""
    
    def setUp(self):
        """Set up test environment."""
        # Event bus for communication
        self.event_bus = EventBus()
        
        # Risk & Portfolio container
        self.risk_portfolio = RiskPortfolioContainer(
            name="TestPortfolio",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )
        
        # Add risk limits
        self.risk_portfolio.add_risk_limit(
            MaxPositionLimit(max_positions=5, name="MaxPositions")
        )
        self.risk_portfolio.add_risk_limit(
            MaxExposureLimit(max_exposure_pct=Decimal("0.8"), name="MaxExposure")
        )
        
        # Signal flow manager
        self.signal_flow = SignalFlowManager(
            event_bus=self.event_bus,
            enable_caching=True,
            enable_validation=True,
            enable_aggregation=False  # No aggregation for clarity
        )
        
        # Backtest broker
        self.broker = BacktestBrokerRefactored(
            initial_cash=Decimal("100000"),
            risk_portfolio_container=self.risk_portfolio
        )
        
        # Execution engine
        self.execution_engine = DefaultExecutionEngine(broker=self.broker)
        
        # Strategy
        self.strategy = SimpleMomentumStrategy(
            lookback_period=5,
            threshold=Decimal("0.03")
        )
        self.strategy.initialize({"symbols": ["AAPL", "GOOGL"]})
        
        # Register strategy with signal flow
        self.signal_flow.register_strategy("momentum", Decimal("1.0"))
        
        # Market data
        self.market_data = {
            "timestamp": datetime.now(),
            "prices": {
                "AAPL": Decimal("150.00"),
                "GOOGL": Decimal("2800.00")
            }
        }
        
        # Event tracking
        self.events_received = []
        self.event_bus.subscribe(EventType.ORDER, self._track_event)
        self.event_bus.subscribe(EventType.FILL, self._track_event)
        self.event_bus.subscribe(EventType.SYSTEM, self._track_event)
    
    def _track_event(self, event: Event):
        """Track events for verification."""
        self.events_received.append(event)
    
    def test_complete_flow_single_signal(self):
        """Test complete flow with single signal."""
        # Create signal
        signal = Signal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={"reason": "strong_momentum"}
        )
        
        async def run_test():
            # 1. Collect signal
            await self.signal_flow.collect_signal(signal)
            
            # 2. Process signals through risk
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            self.assertEqual(len(orders), 1)
            order = orders[0]
            
            # Verify order properties
            self.assertEqual(order.symbol, "AAPL")
            self.assertEqual(order.side, OrderSide.BUY)
            self.assertGreater(order.quantity, 0)
            self.assertGreater(len(order.risk_checks_passed), 0)
            
            # 3. Send order to execution
            exec_order = self.execution_engine.submit_order(order)
            self.assertIsNotNone(exec_order)
            
            # 4. Process order (simulate market)
            fill = self.broker.simulate_fill(
                exec_order,
                market_price=self.market_data["prices"]["AAPL"]
            )
            
            self.assertIsNotNone(fill)
            self.assertEqual(fill.status, FillStatus.FILLED)
            self.assertEqual(fill.filled_quantity, order.quantity)
            
            # 5. Update portfolio with fill
            fill_data = {
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.filled_quantity,
                "price": fill.average_price,
                "timestamp": fill.timestamp,
                "commission": fill.commission
            }
            self.risk_portfolio.update_fills([fill_data])
            
            # 6. Verify portfolio state
            portfolio_state = self.risk_portfolio.get_portfolio_state()
            position = portfolio_state.get_position("AAPL")
            
            self.assertIsNotNone(position)
            self.assertEqual(position.quantity, order.quantity)
            self.assertEqual(position.symbol, "AAPL")
            
            # Verify cash was reduced
            cash_balance = portfolio_state.get_cash_balance()
            expected_cash = Decimal("100000") - (order.quantity * fill.average_price) - fill.commission
            self.assertAlmostEqual(float(cash_balance), float(expected_cash), places=2)
        
        # Run async test
        asyncio.run(run_test())
    
    def test_flow_with_multiple_signals(self):
        """Test flow with multiple signals from different symbols."""
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
                symbol="GOOGL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.7"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        async def run_test():
            # Collect signals
            for signal in signals:
                await self.signal_flow.collect_signal(signal)
            
            # Process signals
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            self.assertEqual(len(orders), 2)
            
            # Process each order
            fills = []
            for order in orders:
                exec_order = self.execution_engine.submit_order(order)
                fill = self.broker.simulate_fill(
                    exec_order,
                    market_price=self.market_data["prices"][order.symbol]
                )
                fills.append(fill)
            
            # Update portfolio with all fills
            fill_data_list = []
            for fill in fills:
                fill_data_list.append({
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.filled_quantity,
                    "price": fill.average_price,
                    "timestamp": fill.timestamp,
                    "commission": fill.commission
                })
            
            self.risk_portfolio.update_fills(fill_data_list)
            
            # Verify both positions exist
            portfolio_state = self.risk_portfolio.get_portfolio_state()
            aapl_position = portfolio_state.get_position("AAPL")
            googl_position = portfolio_state.get_position("GOOGL")
            
            self.assertIsNotNone(aapl_position)
            self.assertIsNotNone(googl_position)
            self.assertGreater(aapl_position.quantity, 0)
            self.assertGreater(googl_position.quantity, 0)
        
        asyncio.run(run_test())
    
    def test_flow_with_exit_signal(self):
        """Test flow with entry followed by exit signal."""
        async def run_test():
            # First create a position
            entry_signal = Signal(
                signal_id="SIG001",
                strategy_id="momentum",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            )
            
            await self.signal_flow.collect_signal(entry_signal)
            
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            # Process entry
            entry_order = orders[0]
            exec_order = self.execution_engine.submit_order(entry_order)
            fill = self.broker.simulate_fill(
                exec_order,
                market_price=self.market_data["prices"]["AAPL"]
            )
            
            self.risk_portfolio.update_fills([{
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.filled_quantity,
                "price": fill.average_price,
                "timestamp": fill.timestamp,
                "commission": fill.commission
            }])
            
            # Verify position exists
            position = self.risk_portfolio.get_portfolio_state().get_position("AAPL")
            self.assertIsNotNone(position)
            entry_quantity = position.quantity
            
            # Now create exit signal
            exit_signal = Signal(
                signal_id="SIG002",
                strategy_id="momentum",
                symbol="AAPL",
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                strength=Decimal("1.0"),
                timestamp=datetime.now() + timedelta(minutes=1),
                metadata={"reason": "momentum_reversal"}
            )
            
            await self.signal_flow.collect_signal(exit_signal)
            
            # Update market price (simulate profit)
            self.market_data["prices"]["AAPL"] = Decimal("155.00")
            
            exit_orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            self.assertEqual(len(exit_orders), 1)
            exit_order = exit_orders[0]
            
            # Process exit
            exec_exit_order = self.execution_engine.submit_order(exit_order)
            exit_fill = self.broker.simulate_fill(
                exec_exit_order,
                market_price=self.market_data["prices"]["AAPL"]
            )
            
            self.risk_portfolio.update_fills([{
                "order_id": exit_fill.order_id,
                "symbol": exit_fill.symbol,
                "side": exit_fill.side.value,
                "quantity": exit_fill.filled_quantity,
                "price": exit_fill.average_price,
                "timestamp": exit_fill.timestamp,
                "commission": exit_fill.commission
            }])
            
            # Verify position is closed
            final_position = self.risk_portfolio.get_portfolio_state().get_position("AAPL")
            self.assertIsNone(final_position)  # Position should be closed
            
            # Verify realized PnL
            metrics = self.risk_portfolio.get_portfolio_state().get_risk_metrics()
            self.assertGreater(metrics.realized_pnl, 0)  # Should have profit
        
        asyncio.run(run_test())
    
    def test_risk_limit_rejection(self):
        """Test that risk limits properly reject signals."""
        # Set very restrictive position limit
        self.risk_portfolio._risk_limits = [
            MaxPositionLimit(max_positions=1, name="MaxPositions")
        ]
        
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
                symbol="GOOGL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        async def run_test():
            # Collect signals
            for signal in signals:
                await self.signal_flow.collect_signal(signal)
            
            # Process signals - should prioritize by strength
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            # Only one order should pass risk limits
            self.assertEqual(len(orders), 1)
            # Higher strength signal should be processed first
            self.assertEqual(orders[0].symbol, "GOOGL")
        
        asyncio.run(run_test())
    
    def test_event_flow(self):
        """Test that events flow correctly through the system."""
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
        
        async def run_test():
            # Clear previous events
            self.events_received.clear()
            
            # Process signal
            await self.signal_flow.collect_signal(signal)
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            # Should have signal collection and order generation events
            system_events = [e for e in self.events_received if e.event_type == EventType.SYSTEM]
            self.assertGreater(len(system_events), 0)
            
            # Check for specific event types
            event_types = [e.payload.get("type") for e in system_events]
            self.assertIn("signal_collected", event_types)
            self.assertIn("order_generated", event_types)
        
        asyncio.run(run_test())


class TestMultiStrategyFlow(unittest.TestCase):
    """Test flow with multiple strategies."""
    
    def setUp(self):
        """Set up test environment."""
        self.event_bus = EventBus()
        self.risk_portfolio = RiskPortfolioContainer(
            name="TestPortfolio",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )
        
        self.signal_flow = SignalFlowManager(
            event_bus=self.event_bus,
            enable_caching=True,
            enable_validation=True,
            enable_aggregation=True,
            aggregation_method="weighted_average"
        )
        
        # Register multiple strategies
        self.signal_flow.register_strategy("momentum", Decimal("1.5"))
        self.signal_flow.register_strategy("mean_reversion", Decimal("1.0"))
        
        self.market_data = {
            "timestamp": datetime.now(),
            "prices": {"AAPL": Decimal("150.00")}
        }
    
    def test_signal_aggregation(self):
        """Test that signals from multiple strategies are aggregated."""
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
        
        async def run_test():
            # Collect signals
            for signal in signals:
                await self.signal_flow.collect_signal(signal)
            
            # Process with aggregation
            orders = await self.signal_flow.process_signals(
                portfolio_state=self.risk_portfolio.get_portfolio_state(),
                position_sizer=self.risk_portfolio._position_sizer,
                risk_limits=self.risk_portfolio._risk_limits,
                market_data=self.market_data
            )
            
            # Should get one aggregated order
            self.assertEqual(len(orders), 1)
            
            # Check aggregated signal metadata
            order = orders[0]
            self.assertIn("aggregation_method", order.source_signal.metadata)
            self.assertIn("source_strategies", order.source_signal.metadata)
            
            # Verify weighted average was applied
            # momentum: 0.8 * 1.5 = 1.2
            # mean_reversion: 0.6 * 1.0 = 0.6
            # weighted avg: (1.2 + 0.6) / (1.5 + 1.0) = 0.72
            expected_strength = Decimal("0.72")
            self.assertAlmostEqual(
                float(order.source_signal.strength),
                float(expected_strength),
                places=2
            )
        
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()