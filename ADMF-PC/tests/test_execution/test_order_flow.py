"""
Tests for order flow through the execution engine.

Tests cover:
- Order creation and validation
- Order lifecycle management
- Event-driven order processing
- Fill generation and handling
- Error cases and edge conditions
"""

import unittest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, call
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.execution.protocols import (
    Order, Fill, OrderStatus, OrderType, OrderSide, FillType
)
from src.execution.execution_engine import DefaultExecutionEngine
from src.execution.order_manager import OrderManager
from src.execution.backtest_broker_refactored import BacktestBrokerRefactored
from src.execution.market_simulation import MarketSimulator
from src.core.events.types import Event, EventType
from src.risk.protocols import PortfolioStateProtocol


class TestOrderCreation(unittest.TestCase):
    """Test order creation and validation."""
    
    def test_create_market_order(self):
        """Test creating a market order."""
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(order.order_id, "TEST_001")
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.quantity, Decimal("100"))
        self.assertIsNone(order.price)
        self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_create_limit_order(self):
        """Test creating a limit order."""
        order = Order(
            order_id="TEST_002",
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            price=Decimal("2800.50"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.price, Decimal("2800.50"))
        self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_create_stop_order(self):
        """Test creating a stop order."""
        order = Order(
            order_id="TEST_003",
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("200"),
            stop_price=Decimal("380.00"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(order.order_type, OrderType.STOP)
        self.assertEqual(order.stop_price, Decimal("380.00"))
        self.assertIsNone(order.price)


class TestOrderManager(unittest.TestCase):
    """Test OrderManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.order_manager = OrderManager()
    
    def test_create_order(self):
        """Test order creation through manager."""
        order_request = {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("100")
        }
        
        order = self.order_manager.create_order(order_request)
        
        self.assertIsNotNone(order.order_id)
        self.assertTrue(order.order_id.startswith("ORD-"))
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_update_order_status(self):
        """Test order status updates."""
        # Use asyncio.run to handle async method
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Track order
            self.order_manager._orders[order.order_id] = order
            
            # Update status
            await self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.SUBMITTED
            )
            
            self.assertEqual(order.status, OrderStatus.SUBMITTED)
            
            # Update to filled
            await self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FILLED,
                filled_quantity=Decimal("100"),
                average_price=Decimal("150.50")
            )
            
            self.assertEqual(order.status, OrderStatus.FILLED)
            self.assertEqual(order.filled_quantity, Decimal("100"))
            self.assertEqual(order.average_price, Decimal("150.50"))
        
        asyncio.run(run_test())
    
    def test_validate_order(self):
        """Test order validation."""
        # Valid order
        valid_order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now()
        )
        
        self.assertTrue(self.order_manager.validate_order(valid_order))
        
        # Invalid order - zero quantity
        invalid_order = Order(
            order_id="TEST_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
            timestamp=datetime.now()
        )
        
        self.assertFalse(self.order_manager.validate_order(invalid_order))
        
        # Invalid order - limit order without price
        invalid_limit = Order(
            order_id="TEST_003",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            timestamp=datetime.now()
        )
        
        self.assertFalse(self.order_manager.validate_order(invalid_limit))


class TestBacktestBrokerRefactored(unittest.TestCase):
    """Test the refactored backtest broker."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock portfolio state
        self.mock_portfolio_state = Mock(spec=PortfolioStateProtocol)
        self.mock_portfolio_state.get_cash_balance.return_value = Decimal("100000")
        self.mock_portfolio_state.get_position.return_value = None
        
        # Create broker
        self.broker = BacktestBrokerRefactored(
            portfolio_state=self.mock_portfolio_state,
            initial_capital=Decimal("100000")
        )
    
    def test_broker_initialization(self):
        """Test broker initialization."""
        self.assertEqual(self.broker.portfolio_state, self.mock_portfolio_state)
        self.assertEqual(len(self.broker._pending_orders), 0)
        self.assertEqual(len(self.broker._order_history), 0)
        self.assertEqual(len(self.broker._fills), 0)
    
    def test_submit_order(self):
        """Test order submission."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Submit order
            result = await self.broker.submit_order(order)
            
            self.assertTrue(result)
            self.assertIn(order.order_id, self.broker._pending_orders)
            self.assertEqual(
                self.broker._pending_orders[order.order_id].status,
                OrderStatus.SUBMITTED
            )
        
        asyncio.run(run_test())
    
    def test_cancel_order(self):
        """Test order cancellation."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Submit order first
            await self.broker.submit_order(order)
            
            # Cancel order
            result = await self.broker.cancel_order(order.order_id)
            
            self.assertTrue(result)
            self.assertNotIn(order.order_id, self.broker._pending_orders)
            self.assertIn(order.order_id, self.broker._order_history)
            self.assertEqual(
                self.broker._order_history[order.order_id].status,
                OrderStatus.CANCELLED
            )
        
        asyncio.run(run_test())
    
    def test_process_fill(self):
        """Test fill processing."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Submit order
            await self.broker.submit_order(order)
            
            # Create fill
            fill = Fill(
                fill_id="FILL_001",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal("100"),
                price=Decimal("150.50"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(),
                fill_type=FillType.COMPLETE
            )
            
            # Process fill
            await self.broker._process_fill(order, fill)
            
            # Check order moved to history
            self.assertNotIn(order.order_id, self.broker._pending_orders)
            self.assertIn(order.order_id, self.broker._order_history)
            
            # Check fill recorded
            self.assertEqual(len(self.broker._fills), 1)
            self.assertEqual(self.broker._fills[0], fill)
            
            # Check portfolio state was updated
            self.mock_portfolio_state.update_position.assert_called_once_with(
                symbol="AAPL",
                quantity_delta=Decimal("100"),
                price=Decimal("150.50"),
                timestamp=fill.timestamp
            )
        
        asyncio.run(run_test())


class TestExecutionEngine(unittest.TestCase):
    """Test the DefaultExecutionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_broker = Mock()
        self.mock_order_manager = Mock()
        self.mock_market_simulator = Mock()
        
        self.engine = DefaultExecutionEngine(
            broker=self.mock_broker,
            order_manager=self.mock_order_manager,
            market_simulator=self.mock_market_simulator
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.broker, self.mock_broker)
        self.assertEqual(self.engine.order_manager, self.mock_order_manager)
        self.assertEqual(self.engine.market_simulator, self.mock_market_simulator)
        self.assertEqual(self.engine._mode, "backtest")
    
    def test_process_order_event(self):
        """Test processing an order event."""
        async def run_test():
            # Create order
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Create event
            event = Event(
                event_type=EventType.ORDER,
                payload={"order": order},
                timestamp=datetime.now()
            )
            
            # Mock order manager validation
            self.mock_order_manager.validate_order.return_value = True
            
            # Mock broker submission
            self.mock_broker.submit_order.return_value = asyncio.Future()
            self.mock_broker.submit_order.return_value.set_result(True)
            
            # Mock market simulator
            fill = Fill(
                fill_id="FILL_001",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=Decimal("150.50"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(),
                fill_type=FillType.COMPLETE
            )
            self.mock_market_simulator.simulate_fill.return_value = asyncio.Future()
            self.mock_market_simulator.simulate_fill.return_value.set_result(fill)
            
            # Process event
            await self.engine.process_event(event)
            
            # Verify flow
            self.mock_order_manager.validate_order.assert_called_once_with(order)
            self.mock_broker.submit_order.assert_called_once_with(order)
            self.mock_market_simulator.simulate_fill.assert_called_once()
        
        asyncio.run(run_test())
    
    def test_process_market_data_event(self):
        """Test processing a market data event."""
        async def run_test():
            # Create market data event
            market_data = {
                "symbol": "AAPL",
                "timestamp": datetime.now(),
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 1000000
            }
            
            event = Event(
                event_type=EventType.BAR,
                payload=market_data,
                timestamp=datetime.now()
            )
            
            # Process event
            await self.engine.process_event(event)
            
            # Check market data cached
            self.assertIn("AAPL", self.engine._market_data_cache)
            self.assertEqual(
                self.engine._market_data_cache["AAPL"]["close"],
                151.0
            )
        
        asyncio.run(run_test())


class TestMarketSimulation(unittest.TestCase):
    """Test market simulation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MarketSimulator(
            slippage_model="percentage",
            slippage_params={"percentage": 0.001},  # 0.1%
            commission_model="percentage",
            commission_params={"percentage": 0.001}  # 0.1%
        )
    
    def test_simulate_market_fill(self):
        """Test market order fill simulation."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "AAPL",
                "bid": 150.00,
                "ask": 150.10,
                "last": 150.05,
                "volume": 1000000
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            
            self.assertIsNotNone(fill)
            self.assertEqual(fill.symbol, "AAPL")
            self.assertEqual(fill.quantity, Decimal("100"))
            
            # Buy order should fill at ask + slippage
            expected_price = Decimal("150.10") * Decimal("1.001")  # ask + 0.1%
            self.assertAlmostEqual(
                float(fill.price),
                float(expected_price),
                places=2
            )
            
            # Commission should be 0.1% of notional
            expected_commission = fill.quantity * fill.price * Decimal("0.001")
            self.assertAlmostEqual(
                float(fill.commission),
                float(expected_commission),
                places=2
            )
        
        asyncio.run(run_test())
    
    def test_simulate_limit_fill(self):
        """Test limit order fill simulation."""
        async def run_test():
            order = Order(
                order_id="TEST_002",
                symbol="GOOGL",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("50"),
                price=Decimal("2800.00"),
                timestamp=datetime.now()
            )
            
            # Market below limit - should not fill
            market_data = {
                "symbol": "GOOGL",
                "bid": 2795.00,
                "ask": 2796.00,
                "last": 2795.50
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
            
            # Market at limit - should fill
            market_data["bid"] = 2800.00
            market_data["ask"] = 2801.00
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNotNone(fill)
            self.assertEqual(fill.price, Decimal("2800.00"))  # Limit price
        
        asyncio.run(run_test())


class TestOrderFlowIntegration(unittest.TestCase):
    """Test complete order flow integration."""
    
    def test_market_order_flow(self):
        """Test complete flow for a market order."""
        async def run_test():
            # Set up components
            portfolio_state = Mock(spec=PortfolioStateProtocol)
            portfolio_state.get_cash_balance.return_value = Decimal("100000")
            portfolio_state.get_position.return_value = None
            
            broker = BacktestBrokerRefactored(
                portfolio_state=portfolio_state,
                initial_capital=Decimal("100000")
            )
            
            order_manager = OrderManager()
            
            market_simulator = MarketSimulator(
                slippage_model="fixed",
                slippage_params={"amount": 0.01},
                commission_model="fixed",
                commission_params={"amount": 1.00}
            )
            
            engine = DefaultExecutionEngine(
                broker=broker,
                order_manager=order_manager,
                market_simulator=market_simulator
            )
            
            # Create and process order
            order_request = {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.MARKET,
                "quantity": Decimal("100")
            }
            
            order = order_manager.create_order(order_request)
            
            # Create order event
            event = Event(
                event_type=EventType.ORDER,
                payload={"order": order},
                timestamp=datetime.now()
            )
            
            # Add market data
            market_event = Event(
                event_type=EventType.BAR,
                payload={
                    "symbol": "AAPL",
                    "bid": 150.00,
                    "ask": 150.10,
                    "last": 150.05
                },
                timestamp=datetime.now()
            )
            
            # Process market data first
            await engine.process_event(market_event)
            
            # Process order
            await engine.process_event(event)
            
            # Verify order was processed
            self.assertEqual(len(broker._order_history), 1)
            filled_order = list(broker._order_history.values())[0]
            self.assertEqual(filled_order.status, OrderStatus.FILLED)
            
            # Verify fill was created
            self.assertEqual(len(broker._fills), 1)
            fill = broker._fills[0]
            self.assertEqual(fill.symbol, "AAPL")
            self.assertEqual(fill.quantity, Decimal("100"))
            
            # Verify portfolio was updated
            portfolio_state.update_position.assert_called_once()
        
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()