"""
Unit tests for Step 2 risk components.

Tests all the core risk management components created for Step 2:
- PortfolioState and Position tracking
- RiskLimits enforcement
- PositionSizer calculations
- OrderManager functionality
- RiskContainer integration

Architecture Context:
    - Part of: Step 2 - Add Risk Container testing
    - Validates: Protocol-based risk components work correctly
    - Coverage: Unit-level testing of individual components
    - Dependencies: Core components, risk models
"""

import unittest
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime

# Import Step 2 components
from src.risk.step2_portfolio_state import PortfolioState, Position
from src.risk.step2_risk_limits import RiskLimits
from src.risk.step2_position_sizer import PositionSizer
from src.risk.step2_order_manager import OrderManager
from src.risk.risk_container import RiskContainer
from src.risk.step2_container_factory import create_test_risk_container
from src.risk.models import (
    RiskConfig, TradingSignal, Order, Fill, 
    SignalType, OrderSide, OrderType
)


class TestPosition(unittest.TestCase):
    """Test Position class functionality."""
    
    def setUp(self):
        """Set up test position."""
        self.position = Position("SPY", Decimal('0'), Decimal('0'))
    
    def test_position_initialization(self):
        """Test position initialization."""
        self.assertEqual(self.position.symbol, "SPY")
        self.assertEqual(self.position.quantity, Decimal('0'))
        self.assertEqual(self.position.avg_price, Decimal('0'))
        self.assertEqual(self.position.realized_pnl, Decimal('0'))
    
    def test_opening_long_position(self):
        """Test opening a long position."""
        fill = Fill(
            fill_id="FILL001",
            order_id="ORD001",
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        
        self.position.update_from_fill(fill)
        
        self.assertEqual(self.position.quantity, Decimal('100'))
        self.assertEqual(self.position.avg_price, Decimal('400.00'))
        self.assertEqual(self.position.realized_pnl, Decimal('0'))
    
    def test_adding_to_long_position(self):
        """Test adding to existing long position."""
        # Open position
        fill1 = Fill(
            fill_id="FILL001", order_id="ORD001", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        self.position.update_from_fill(fill1)
        
        # Add to position
        fill2 = Fill(
            fill_id="FILL002", order_id="ORD002", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('50'), price=Decimal('410.00'),
            timestamp=datetime.now()
        )
        self.position.update_from_fill(fill2)
        
        self.assertEqual(self.position.quantity, Decimal('150'))
        # Average price: (100*400 + 50*410) / 150 = 403.33
        self.assertAlmostEqual(float(self.position.avg_price), 403.33, places=2)
    
    def test_partial_close_position(self):
        """Test partially closing a position."""
        # Open position
        fill1 = Fill(
            fill_id="FILL001", order_id="ORD001", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        self.position.update_from_fill(fill1)
        
        # Partial close at higher price
        fill2 = Fill(
            fill_id="FILL002", order_id="ORD002", symbol="SPY",
            side=OrderSide.SELL, quantity=Decimal('30'), price=Decimal('420.00'),
            timestamp=datetime.now()
        )
        self.position.update_from_fill(fill2)
        
        self.assertEqual(self.position.quantity, Decimal('70'))
        self.assertEqual(self.position.avg_price, Decimal('400.00'))  # Unchanged
        # Realized PnL: 30 * (420 - 400) = 600
        self.assertEqual(self.position.realized_pnl, Decimal('600'))
    
    def test_market_value_calculation(self):
        """Test market value calculation."""
        self.position.quantity = Decimal('100')
        self.position.avg_price = Decimal('400.00')
        
        market_value = self.position.get_market_value(420.0)
        self.assertEqual(market_value, Decimal('42000'))
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        self.position.quantity = Decimal('100')
        self.position.avg_price = Decimal('400.00')
        
        unrealized_pnl = self.position.get_unrealized_pnl(420.0)
        self.assertEqual(unrealized_pnl, Decimal('2000'))  # 100 * (420 - 400)


class TestPortfolioState(unittest.TestCase):
    """Test PortfolioState functionality."""
    
    def setUp(self):
        """Set up test portfolio."""
        self.portfolio = PortfolioState("test_container", 100000.0)
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.container_id, "test_container")
        self.assertEqual(self.portfolio.cash, Decimal('100000'))
        self.assertEqual(self.portfolio.total_value, Decimal('100000'))
        self.assertEqual(len(self.portfolio.positions), 0)
    
    def test_position_update_from_fill(self):
        """Test updating positions from fills."""
        fill = Fill(
            fill_id="FILL001", order_id="ORD001", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        
        self.portfolio.update_position(fill)
        
        # Check position created
        self.assertIn("SPY", self.portfolio.positions)
        position = self.portfolio.positions["SPY"]
        self.assertEqual(position.quantity, Decimal('100'))
        self.assertEqual(position.avg_price, Decimal('400.00'))
        
        # Check cash updated
        expected_cash = Decimal('100000') - Decimal('40000')  # 100 * 400
        self.assertEqual(self.portfolio.cash, expected_cash)
    
    def test_portfolio_value_calculation(self):
        """Test total portfolio value calculation."""
        # Add position
        fill = Fill(
            fill_id="FILL001", order_id="ORD001", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        self.portfolio.update_position(fill)
        
        # Update market prices
        self.portfolio.update_prices({"SPY": 420.0})
        
        # Calculate total value
        total_value = self.portfolio.calculate_total_value()
        expected_value = Decimal('60000') + Decimal('42000')  # cash + position value
        self.assertEqual(total_value, expected_value)
    
    def test_exposure_calculation(self):
        """Test exposure metrics calculation."""
        # Add long position
        fill1 = Fill(
            fill_id="FILL001", order_id="ORD001", symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400.00'),
            timestamp=datetime.now()
        )
        self.portfolio.update_position(fill1)
        
        # Update prices
        self.portfolio.update_prices({"SPY": 420.0})
        
        exposure = self.portfolio.get_exposure()
        
        self.assertEqual(exposure['long_exposure'], 42000.0)  # 100 * 420
        self.assertEqual(exposure['short_exposure'], 0.0)
        self.assertEqual(exposure['gross_exposure'], 42000.0)
        self.assertEqual(exposure['net_exposure'], 42000.0)


class TestRiskLimits(unittest.TestCase):
    """Test RiskLimits functionality."""
    
    def setUp(self):
        """Set up test risk limits."""
        config = RiskConfig(
            initial_capital=100000.0,
            max_position_size=0.1,  # 10%
            max_portfolio_risk=0.02,  # 2%
            max_drawdown=0.2,  # 20%
            max_concentration=0.5  # 50%
        )
        self.risk_limits = RiskLimits(config)
        self.portfolio = PortfolioState("test", 100000.0)
    
    def test_risk_limits_initialization(self):
        """Test risk limits initialization."""
        self.assertEqual(self.risk_limits.max_position_size, 0.1)
        self.assertEqual(self.risk_limits.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_limits.max_drawdown, 0.2)
    
    def test_position_limit_check_passes(self):
        """Test position limit check passes for small position."""
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        # Small position should pass
        self.portfolio.update_prices({"SPY": 400.0})
        result = self.risk_limits.can_trade(self.portfolio, signal)
        self.assertTrue(result)
    
    def test_portfolio_risk_check(self):
        """Test portfolio risk check."""
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.5')  # Should pass (< 0.2 scaled)
        )
        
        result = self.risk_limits.can_trade(self.portfolio, signal)
        self.assertTrue(result)
        
        # High strength signal should fail
        signal.strength = Decimal('0.8')  # Should fail (> 0.2 scaled)
        result = self.risk_limits.can_trade(self.portfolio, signal)
        self.assertFalse(result)


class TestPositionSizer(unittest.TestCase):
    """Test PositionSizer functionality."""
    
    def setUp(self):
        """Set up test position sizer."""
        config = RiskConfig(
            initial_capital=100000.0,
            sizing_method='fixed',
            fixed_position_size=1000.0,
            percent_risk_per_trade=0.01,
            default_stop_loss_pct=0.05
        )
        self.sizer = PositionSizer('fixed', config)
        self.portfolio = PortfolioState("test", 100000.0)
    
    def test_fixed_sizing(self):
        """Test fixed dollar amount sizing."""
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('1.0')
        )
        
        self.portfolio.update_prices({"SPY": 400.0})
        size = self.sizer.calculate_size(signal, self.portfolio)
        
        # $1000 / $400 = 2.5 shares, rounded to 2
        expected_size = Decimal('2')  # Quantized to whole shares
        self.assertEqual(size, expected_size)
    
    def test_percent_risk_sizing(self):
        """Test percentage risk sizing."""
        config = RiskConfig(
            initial_capital=100000.0,
            sizing_method='percent_risk',
            percent_risk_per_trade=0.01,
            default_stop_loss_pct=0.05
        )
        sizer = PositionSizer('percent_risk', config)
        
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('1.0')
        )
        
        self.portfolio.update_prices({"SPY": 400.0})
        size = sizer.calculate_size(signal, self.portfolio)
        
        # Risk amount: $100,000 * 0.01 = $1,000
        # Stop distance: $400 * 0.05 = $20
        # Size: $1,000 / $20 = 50 shares
        expected_size = Decimal('50')
        self.assertEqual(size, expected_size)
    
    def test_signal_strength_scaling(self):
        """Test signal strength affects position size."""
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.5')  # Half strength
        )
        
        self.portfolio.update_prices({"SPY": 400.0})
        size = self.sizer.calculate_size(signal, self.portfolio)
        
        # Base size would be 2, scaled by 0.5 = 1.0, rounded to 1
        expected_size = Decimal('1')
        self.assertEqual(size, expected_size)


class TestOrderManager(unittest.TestCase):
    """Test OrderManager functionality."""
    
    def setUp(self):
        """Set up test order manager."""
        self.order_manager = OrderManager("test_container")
    
    def test_order_manager_initialization(self):
        """Test order manager initialization."""
        self.assertEqual(self.order_manager.container_id, "test_container")
        self.assertEqual(len(self.order_manager.created_orders), 0)
        self.assertEqual(self.order_manager.order_count, 0)
    
    def test_order_creation(self):
        """Test order creation from signal."""
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        current_prices = {"SPY": 400.0}
        size = Decimal('100')
        
        order = self.order_manager.create_order(signal, size, current_prices)
        
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, "SPY")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, Decimal('100'))
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(len(self.order_manager.created_orders), 1)
    
    def test_order_id_generation(self):
        """Test unique order ID generation."""
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        
        current_prices = {"SPY": 400.0}
        
        order1 = self.order_manager.create_order(signal, Decimal('100'), current_prices)
        order2 = self.order_manager.create_order(signal, Decimal('50'), current_prices)
        
        self.assertNotEqual(order1.order_id, order2.order_id)
        self.assertTrue(order1.order_id.startswith("ORD_test_container_"))
        self.assertTrue(order2.order_id.startswith("ORD_test_container_"))
    
    def test_order_stats(self):
        """Test order statistics calculation."""
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        
        current_prices = {"SPY": 400.0}
        
        # Create some orders
        self.order_manager.create_order(signal, Decimal('100'), current_prices)
        signal.side = OrderSide.SELL
        self.order_manager.create_order(signal, Decimal('50'), current_prices)
        
        stats = self.order_manager.get_order_stats()
        
        self.assertEqual(stats['total_orders'], 2)
        self.assertEqual(stats['orders_by_side']['BUY'], 1)
        self.assertEqual(stats['orders_by_side']['SELL'], 1)
        self.assertEqual(stats['orders_by_symbol']['SPY'], 2)
        self.assertEqual(stats['avg_order_size'], 75.0)  # (100 + 50) / 2


class TestRiskContainerFactory(unittest.TestCase):
    """Test risk container factory functions."""
    
    def test_test_container_creation(self):
        """Test test container creation."""
        container = create_test_risk_container("test_001", 10000.0, "fixed")
        
        self.assertIsInstance(container, RiskContainer)
        self.assertEqual(container.container_id, "test_001")
        self.assertEqual(container.config.initial_capital, 10000.0)
        self.assertEqual(container.config.sizing_method, "fixed")
    
    def test_container_components_initialization(self):
        """Test that all container components are properly initialized."""
        container = create_test_risk_container("test_001", 10000.0)
        
        # Check all components exist
        self.assertIsNotNone(container.portfolio_state)
        self.assertIsNotNone(container.position_sizer)
        self.assertIsNotNone(container.risk_limits)
        self.assertIsNotNone(container.order_manager)
        self.assertIsNotNone(container.event_bus)


class TestRiskContainerIntegration(unittest.TestCase):
    """Test RiskContainer integration."""
    
    def setUp(self):
        """Set up test risk container."""
        self.container = create_test_risk_container("integration_test", 50000.0, "fixed")
    
    def test_signal_processing_flow(self):
        """Test complete signal processing flow."""
        # Update market data
        self.container.update_market_data({"SPY": 400.0})
        
        # Create trading signal
        signal = TradingSignal(
            signal_id="SIG001",
            strategy_id="momentum",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.8')
        )
        
        # Process signal
        initial_orders = self.container.created_orders
        self.container.on_signal(signal)
        
        # Check order was created
        self.assertEqual(self.container.created_orders, initial_orders + 1)
        self.assertEqual(self.container.processed_signals, 1)
    
    def test_fill_processing(self):
        """Test fill processing updates portfolio."""
        # Update market data
        self.container.update_market_data({"SPY": 400.0})
        
        # Create and process signal
        signal = TradingSignal(
            signal_id="SIG001", strategy_id="momentum", symbol="SPY",
            signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8')
        )
        self.container.on_signal(signal)
        
        # Create fill
        fill = Fill(
            fill_id="FILL001",
            order_id="ORD_integration_test_0001",
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('400.0'),
            timestamp=datetime.now()
        )
        
        initial_cash = self.container.portfolio_state.cash
        self.container.on_fill(fill)
        
        # Check position was created
        self.assertIn("SPY", self.container.portfolio_state.positions)
        position = self.container.portfolio_state.positions["SPY"]
        self.assertEqual(position.quantity, Decimal('100'))
        
        # Check cash was reduced
        expected_cash = initial_cash - Decimal('40000')  # 100 * 400
        self.assertEqual(self.container.portfolio_state.cash, expected_cash)
    
    def test_container_state_reporting(self):
        """Test container state reporting."""
        state = self.container.get_state()
        
        self.assertIn('container_id', state)
        self.assertIn('processed_signals', state)
        self.assertIn('portfolio_value', state)
        self.assertIn('risk_config', state)
        
        self.assertEqual(state['container_id'], 'integration_test')
        self.assertEqual(state['processed_signals'], 0)
        self.assertEqual(state['portfolio_value'], 50000.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)