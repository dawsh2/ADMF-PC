"""
Unit tests for risk limit implementations.
Tests all risk limits including position, drawdown, VaR, exposure,
concentration, leverage, daily loss, and symbol restrictions.
"""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock
from typing import Optional
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.risk.risk_limits import (
    MaxPositionLimit,
    MaxDrawdownLimit,
    VaRLimit,
    MaxExposureLimit,
    ConcentrationLimit,
    LeverageLimit,
    DailyLossLimit,
    SymbolRestrictionLimit
)
from src.risk.protocols import Order, OrderSide, OrderType, Position, RiskMetrics, Signal, SignalType
from datetime import datetime
def create_test_order(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    order_type: OrderType = OrderType.MARKET,
    price: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None,
    time_in_force: str = "GTC"
) -> Order:
    """Helper to create test orders with all required fields."""
    # Create a dummy signal
    test_signal = Signal(
        signal_id=f"signal_{order_id}",
        strategy_id="test_strategy",
        symbol=symbol,
        signal_type=SignalType.ENTRY if side == OrderSide.BUY else SignalType.EXIT,
        side=side,
        strength=Decimal("1.0"),
        timestamp=datetime.now(),
        metadata={}
    )
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        source_signal=test_signal,
        risk_checks_passed=[],
        timestamp=datetime.now(),
        metadata={}
    )
class MockPortfolioState:
    """Mock portfolio state for testing."""
    def __init__(
        self,
        total_value=Decimal("100000"),
        cash_balance=Decimal("50000"),
        positions=None,
        risk_metrics=None
    ):
        self.total_value = total_value
        self.cash_balance = cash_balance
        self.positions = positions or {}
        self._risk_metrics = risk_metrics or RiskMetrics(
            total_value=total_value,
            positions_value=Decimal("50000"),
            cash_balance=cash_balance,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            current_drawdown=Decimal("0"),
            max_drawdown=Decimal("0"),
            sharpe_ratio=None,
            var_95=None,
            leverage=Decimal("1.0"),
            concentration={},
            timestamp=datetime.now()
        )
    def get_total_value(self):
        return self.total_value
    def get_cash_balance(self):
        return self.cash_balance
    def get_position(self, symbol):
        return self.positions.get(symbol)
    def get_risk_metrics(self):
        return self._risk_metrics
class TestMaxPositionLimit(unittest.TestCase):
    """Test MaxPositionLimit functionality."""
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = MockPortfolioState()
        self.market_data = {"prices": {"AAPL": 150.00}}
    def test_initialization(self):
        """Test limit initialization."""
        # Test with value limit
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        self.assertEqual(limit.max_position_value, Decimal("10000"))
        self.assertIsNone(limit.max_position_percent)
        # Test with percentage limit
        limit = MaxPositionLimit(max_position_percent=Decimal("0.10"))
        self.assertIsNone(limit.max_position_value)
        self.assertEqual(limit.max_position_percent, Decimal("0.10"))
        # Test with both
        limit = MaxPositionLimit(
            max_position_value=Decimal("10000"),
            max_position_percent=Decimal("0.10")
        )
        self.assertEqual(limit.max_position_value, Decimal("10000"))
        self.assertEqual(limit.max_position_percent, Decimal("0.10"))
        # Test error on neither
        with self.assertRaises(ValueError):
            MaxPositionLimit()
    def test_value_limit_check(self):
        """Test position value limit checking."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        # Order within limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),  # 50 * 150 = 7500
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertTrue(passes)
        self.assertIsNone(reason)
        # Order exceeding limit
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("exceeds limit", reason)
    def test_percentage_limit_check(self):
        """Test position percentage limit checking."""
        limit = MaxPositionLimit(max_position_percent=Decimal("0.10"))  # 10%
        # Order within limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),  # 50 * 150 = 7500 = 7.5% of 100k
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertTrue(passes)
        # Order exceeding limit
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000 = 15% of 100k
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("15.0%", reason)
        self.assertIn("exceeds limit 10.0%", reason)
    def test_existing_position_consideration(self):
        """Test that existing positions are considered."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        # Add existing position
        existing_position = Position(
            symbol="AAPL",
            quantity=Decimal("30"),
            average_price=Decimal("145"),
            current_price=Decimal("150"),
            unrealized_pnl=Decimal("150"),
            realized_pnl=Decimal("0"),
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={}
        )
        self.portfolio.positions["AAPL"] = existing_position
        # Order that would exceed limit with existing position
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),  # Total: 80 * 150 = 12000 > 10000
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("12000", reason)
    def test_sell_order_reduces_position(self):
        """Test that sell orders reduce position size."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        # Add large existing position
        existing_position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_price=Decimal("145"),
            current_price=Decimal("150"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={}
        )
        self.portfolio.positions["AAPL"] = existing_position
        # Sell order should pass
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertTrue(passes)
    def test_missing_price_handling(self):
        """Test handling of missing price data."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        order = create_test_order(
            order_id="test_001",
            symbol="UNKNOWN",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("No price available", reason)
    def test_limit_order_with_price(self):
        """Test limit order with specified price."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.LIMIT,
            price=Decimal("155"),  # 50 * 155 = 7750
        )
        passes, reason = limit.check_limit(order, self.portfolio, self.market_data)
        self.assertTrue(passes)
    def test_violation_recording(self):
        """Test that violations are recorded."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        # Create violating order
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000
            order_type=OrderType.MARKET,
            price=None
        )
        limit.check_limit(order, self.portfolio, self.market_data)
        violations = limit.get_violations()
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["order_id"], "test_001")
        self.assertEqual(violations[0]["symbol"], "AAPL")
        self.assertIn("exceeds limit", violations[0]["reason"])
    def test_get_limit_info(self):
        """Test limit info retrieval."""
        limit = MaxPositionLimit(
            max_position_value=Decimal("10000"),
            max_position_percent=Decimal("0.10")
        )
        info = limit.get_limit_info()
        self.assertEqual(info["name"], "MaxPositionLimit")
        self.assertEqual(info["max_position_value"], "10000")
        self.assertEqual(info["max_position_percent"], "0.10")
        self.assertEqual(info["violations_count"], 0)
class TestMaxDrawdownLimit(unittest.TestCase):
    """Test MaxDrawdownLimit functionality."""
    def setUp(self):
        """Set up test fixtures."""
        self.market_data = {"prices": {"AAPL": 150.00}}
    def test_initialization(self):
        """Test limit initialization."""
        limit = MaxDrawdownLimit(max_drawdown=Decimal("0.20"))
        self.assertEqual(limit.max_drawdown, Decimal("0.20"))
        self.assertIsNone(limit.lookback_days)
        limit = MaxDrawdownLimit(
            max_drawdown=Decimal("0.15"),
            lookback_days=30
        )
        self.assertEqual(limit.max_drawdown, Decimal("0.15"))
        self.assertEqual(limit.lookback_days, 30)
    def test_drawdown_within_limit(self):
        """Test when drawdown is within limit."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("-5000"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0.10"),  # 10% drawdown
                max_drawdown=Decimal("0.10"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = MaxDrawdownLimit(max_drawdown=Decimal("0.20"))  # 20% limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, self.market_data)
        self.assertTrue(passes)
        self.assertIsNone(reason)
    def test_drawdown_exceeds_limit(self):
        """Test when drawdown exceeds limit."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("-15000"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0.25"),  # 25% drawdown
                max_drawdown=Decimal("0.25"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = MaxDrawdownLimit(max_drawdown=Decimal("0.20"))  # 20% limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("25.0%", reason)
        self.assertIn("exceeds limit 20.0%", reason)
class TestVaRLimit(unittest.TestCase):
    """Test VaRLimit functionality."""
    def setUp(self):
        """Set up test fixtures."""
        self.market_data = {"prices": {"AAPL": 150.00}}
    def test_initialization(self):
        """Test limit initialization."""
        limit = VaRLimit(max_var=Decimal("0.05"))
        self.assertEqual(limit.max_var, Decimal("0.05"))
        self.assertEqual(limit.confidence_level, Decimal("0.95"))
        limit = VaRLimit(
            max_var=Decimal("0.03"),
            confidence_level=Decimal("0.99")
        )
        self.assertEqual(limit.max_var, Decimal("0.03"))
        self.assertEqual(limit.confidence_level, Decimal("0.99"))
    def test_var_within_limit(self):
        """Test when VaR is within limit."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=Decimal("3000"),  # 3% of portfolio
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = VaRLimit(max_var=Decimal("0.05"))  # 5% limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, self.market_data)
        self.assertTrue(passes)
    def test_var_exceeds_limit(self):
        """Test when VaR exceeds limit."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=Decimal("6000"),  # 6% of portfolio
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = VaRLimit(max_var=Decimal("0.05"))  # 5% limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, self.market_data)
        self.assertFalse(passes)
        self.assertIn("6.0%", reason)
        self.assertIn("exceeds limit 5.0%", reason)
    def test_var_not_available(self):
        """Test when VaR is not available."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,  # VaR not calculated
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = VaRLimit(max_var=Decimal("0.05"))
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        # Should pass when VaR not available
        passes, reason = limit.check_limit(order, portfolio, self.market_data)
        self.assertTrue(passes)
class TestMaxExposureLimit(unittest.TestCase):
    """Test MaxExposureLimit functionality."""
    def test_exposure_calculation(self):
        """Test exposure calculation for buy orders."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("40000"),  # 40% exposure
                cash_balance=Decimal("60000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = MaxExposureLimit(max_exposure_pct=Decimal("80"))  # 80% max
        # Buy order that would bring exposure to 55%
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # Buy order that would exceed limit
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("300"),  # 300 * 150 = 45000, total 85%
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("85.0%", reason)
        self.assertIn("exceed limit 80.0%", reason)
    def test_sell_order_reduces_exposure(self):
        """Test that sell orders reduce exposure."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("85000"),  # 85% exposure
                cash_balance=Decimal("15000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = MaxExposureLimit(max_exposure_pct=Decimal("80"))
        # Sell order should always pass
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
class TestConcentrationLimit(unittest.TestCase):
    """Test ConcentrationLimit functionality."""
    def test_single_position_concentration(self):
        """Test single position concentration limit."""
        portfolio = MockPortfolioState()
        # Add existing position
        existing_position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_price=Decimal("145"),
            current_price=Decimal("150"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={}
        )
        portfolio.positions["AAPL"] = existing_position
        limit = ConcentrationLimit(max_single_position=Decimal("0.25"))  # 25% max
        # Order that would bring position to 22.5%
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),  # 50 * 150 = 7500
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # Order that would exceed limit
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # Total would be 30%
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("30.0%", reason)
        self.assertIn("exceeds limit 25.0%", reason)
    def test_new_position_concentration(self):
        """Test concentration limit for new position."""
        portfolio = MockPortfolioState()
        limit = ConcentrationLimit(max_single_position=Decimal("0.20"))  # 20% max
        # New position within limit
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000 = 15%
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # New position exceeding limit
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("150"),  # 150 * 150 = 22500 = 22.5%
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
class TestLeverageLimit(unittest.TestCase):
    """Test LeverageLimit functionality."""
    def test_no_leverage_scenario(self):
        """Test when no leverage is used."""
        portfolio = MockPortfolioState(
            cash_balance=Decimal("60000"),
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("40000"),
                cash_balance=Decimal("60000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = LeverageLimit(max_leverage=Decimal("2.0"))
        # Order within cash balance
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # 100 * 150 = 15000 < 60000 cash
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
    def test_leverage_required(self):
        """Test when leverage would be required."""
        portfolio = MockPortfolioState(
            cash_balance=Decimal("10000"),
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("90000"),
                cash_balance=Decimal("10000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = LeverageLimit(max_leverage=Decimal("1.5"))
        # Order requiring leverage
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("200"),  # 200 * 150 = 30000 > 10000 cash
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        # New positions value would be 120000, leverage = 1.2x
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # Order requiring too much leverage
        order = create_test_order(
            order_id="test_002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("450"),  # Would require >1.5x leverage
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("x exceeds limit", reason)
    def test_sell_order_always_passes(self):
        """Test that sell orders always pass leverage check."""
        portfolio = MockPortfolioState(
            cash_balance=Decimal("0"),
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("100000"),
                cash_balance=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = LeverageLimit(max_leverage=Decimal("1.0"))
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
class TestDailyLossLimit(unittest.TestCase):
    """Test DailyLossLimit functionality."""
    def test_no_loss_scenario(self):
        """Test when there's no daily loss."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("1000"),  # Profit
                realized_pnl=Decimal("500"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = DailyLossLimit(max_daily_loss=Decimal("0.02"))  # 2% max loss
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
    def test_daily_loss_exceeded(self):
        """Test when daily loss exceeds limit."""
        portfolio = MockPortfolioState(
            risk_metrics=RiskMetrics(
                total_value=Decimal("100000"),
                positions_value=Decimal("50000"),
                cash_balance=Decimal("50000"),
                unrealized_pnl=Decimal("-2000"),  # -2%
                realized_pnl=Decimal("-1000"),     # -1%
                current_drawdown=Decimal("0.03"),
                max_drawdown=Decimal("0.03"),
                sharpe_ratio=None,
                var_95=None,
                leverage=Decimal("1.0"),
                concentration={},
                timestamp=datetime.now()
            )
        )
        limit = DailyLossLimit(max_daily_loss=Decimal("0.02"))  # 2% max loss
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("3.0%", reason)
        self.assertIn("exceeds limit 2.0%", reason)
class TestSymbolRestrictionLimit(unittest.TestCase):
    """Test SymbolRestrictionLimit functionality."""
    def test_allowed_symbols_only(self):
        """Test restriction to allowed symbols only."""
        limit = SymbolRestrictionLimit(
            allowed_symbols={"AAPL", "GOOGL", "MSFT"}
        )
        portfolio = MockPortfolioState()
        market_data = {}
        # Allowed symbol
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # Not allowed symbol
        order = create_test_order(
            order_id="test_002",
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("not in allowed list", reason)
    def test_blocked_symbols(self):
        """Test blocking specific symbols."""
        limit = SymbolRestrictionLimit(
            blocked_symbols={"PENNY1", "PENNY2", "RISKY"}
        )
        portfolio = MockPortfolioState()
        market_data = {}
        # Non-blocked symbol
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
        # Blocked symbol
        order = create_test_order(
            order_id="test_002",
            symbol="PENNY1",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("blocked for trading", reason)
    def test_combined_restrictions(self):
        """Test combined allowed and blocked lists."""
        limit = SymbolRestrictionLimit(
            allowed_symbols={"AAPL", "GOOGL", "MSFT", "RISKY"},
            blocked_symbols={"RISKY"}  # Block overrides allow
        )
        portfolio = MockPortfolioState()
        market_data = {}
        # Blocked takes precedence
        order = create_test_order(
            order_id="test_001",
            symbol="RISKY",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertFalse(passes)
        self.assertIn("blocked for trading", reason)
    def test_no_restrictions(self):
        """Test with no restrictions (all allowed)."""
        limit = SymbolRestrictionLimit()
        portfolio = MockPortfolioState()
        market_data = {}
        # Any symbol should pass
        order = create_test_order(
            order_id="test_001",
            symbol="RANDOM",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        passes, reason = limit.check_limit(order, portfolio, market_data)
        self.assertTrue(passes)
    def test_get_limit_info(self):
        """Test limit info retrieval."""
        limit = SymbolRestrictionLimit(
            allowed_symbols={"AAPL", "GOOGL"},
            blocked_symbols={"PENNY1", "PENNY2"}
        )
        info = limit.get_limit_info()
        self.assertEqual(info["name"], "SymbolRestrictionLimit")
        self.assertEqual(info["allowed_symbols_count"], 2)
        self.assertEqual(set(info["blocked_symbols"]), {"PENNY1", "PENNY2"})
class TestRiskLimitEdgeCases(unittest.TestCase):
    """Test edge cases across all risk limits."""
    def test_zero_portfolio_value(self):
        """Test limits with zero portfolio value."""
        portfolio = MockPortfolioState(
            total_value=Decimal("0"),
            cash_balance=Decimal("0")
        )
        order = create_test_order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            price=None
        )
        market_data = {"prices": {"AAPL": 150.00}}
        # Test percentage-based limits
        limits = [
            MaxPositionLimit(max_position_percent=Decimal("0.10")),
            ConcentrationLimit(max_single_position=Decimal("0.25")),
            MaxExposureLimit(max_exposure_pct=Decimal("80"))
        ]
        for limit in limits:
            passes, reason = limit.check_limit(order, portfolio, market_data)
            # Should handle gracefully, exact behavior depends on implementation
            self.assertIsInstance(passes, bool)
    def test_multiple_violations(self):
        """Test recording multiple violations."""
        limit = MaxPositionLimit(max_position_value=Decimal("10000"))
        portfolio = MockPortfolioState()
        market_data = {"prices": {"AAPL": 150.00}}
        # Generate multiple violations
        for i in range(150):
            order = create_test_order(
                order_id=f"test_{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("100"),  # Always violates
                order_type=OrderType.MARKET,
                price=None
            )
            limit.check_limit(order, portfolio, market_data)
        # Should only keep last 100 violations
        violations = limit.get_violations()
        self.assertEqual(len(violations), 100)
        self.assertEqual(violations[0]["order_id"], "test_050")
        self.assertEqual(violations[-1]["order_id"], "test_149")
if __name__ == "__main__":
    unittest.main()
