"""
Comprehensive tests for the UnifiedBacktestEngine.

Tests cover:
- Basic initialization
- Component integration
- Signal processing flow
- Order execution
- Portfolio state management
- Event system integration
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.execution import UnifiedBacktestEngine, BacktestConfig, BacktestResults
from src.risk.protocols import Signal, SignalType, OrderSide, Order
from src.risk.risk_portfolio import RiskPortfolioContainer
from src.execution.execution_engine import DefaultExecutionEngine
from src.execution.backtest_broker_refactored import BacktestBrokerRefactored


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating a backtest configuration."""
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            symbols=["AAPL", "GOOGL"],
            frequency="1d",
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
        
        self.assertEqual(config.start_date, datetime(2023, 1, 1))
        self.assertEqual(config.end_date, datetime(2023, 12, 31))
        self.assertEqual(config.initial_capital, Decimal("100000"))
        self.assertEqual(config.symbols, ["AAPL", "GOOGL"])
        self.assertEqual(config.commission, Decimal("0.001"))
        self.assertEqual(config.slippage, Decimal("0.0005"))
        self.assertTrue(config.enable_shorting)
        self.assertTrue(config.use_adjusted_close)
    
    def test_config_defaults(self):
        """Test default values in config."""
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("50000"),
            symbols=["SPY"]
        )
        
        self.assertEqual(config.frequency, "1d")
        self.assertEqual(config.commission, Decimal("0.001"))
        self.assertEqual(config.slippage, Decimal("0.0005"))
        self.assertIsNone(config.benchmark_symbol)


class TestUnifiedBacktestEngine(unittest.TestCase):
    """Test UnifiedBacktestEngine initialization and setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_capital=Decimal("100000"),
            symbols=["TEST"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
    
    def test_engine_initialization(self):
        """Test engine initializes all components correctly."""
        engine = UnifiedBacktestEngine(self.config)
        
        # Check core components exist
        self.assertIsInstance(engine.risk_portfolio, RiskPortfolioContainer)
        self.assertIsInstance(engine.broker, BacktestBrokerRefactored)
        self.assertIsInstance(engine.execution_engine, DefaultExecutionEngine)
        self.assertIsNotNone(engine.market_simulator)
        self.assertIsNotNone(engine.order_manager)
        
        # Check initialization values
        self.assertEqual(engine.config, self.config)
        self.assertEqual(engine.risk_portfolio.name, "backtest_portfolio")
        
    def test_engine_with_container(self):
        """Test engine initialization with container."""
        mock_container = Mock()
        mock_container.event_bus = Mock()
        
        engine = UnifiedBacktestEngine(self.config, container=mock_container)
        
        self.assertEqual(engine.container, mock_container)
    
    def test_broker_uses_portfolio_state(self):
        """Test that broker uses risk module's portfolio state."""
        engine = UnifiedBacktestEngine(self.config)
        
        # Get portfolio states
        risk_portfolio_state = engine.risk_portfolio.get_portfolio_state()
        broker_portfolio_state = engine.broker.portfolio_state
        
        # They should be the same object (single source of truth)
        self.assertIs(broker_portfolio_state, risk_portfolio_state)
        
        # Check initial capital is consistent
        self.assertEqual(
            risk_portfolio_state.get_cash_balance(),
            self.config.initial_capital
        )


class TestSignalProcessing(unittest.TestCase):
    """Test signal processing through the engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_capital=Decimal("100000"),
            symbols=["AAPL", "GOOGL"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
        self.engine = UnifiedBacktestEngine(self.config)
        
    def test_generate_signals_single(self):
        """Test signal generation from strategy with single signal."""
        # Mock strategy with generate_signal method
        strategy = Mock()
        signal = Signal(
            signal_id="TEST_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        strategy.generate_signal.return_value = signal
        
        market_data = {"timestamp": datetime.now(), "prices": {"AAPL": 150.0}}
        signals = self.engine._generate_signals(strategy, market_data)
        
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0], signal)
        strategy.generate_signal.assert_called_once_with(market_data)
    
    def test_generate_signals_multiple(self):
        """Test signal generation from strategy with multiple signals."""
        # Mock strategy with generate_signals method
        strategy = Mock()
        signals = [
            Signal(
                signal_id="TEST_001",
                strategy_id="test_strategy",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),
                metadata={}
            ),
            Signal(
                signal_id="TEST_002",
                strategy_id="test_strategy",
                symbol="GOOGL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.7"),
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        strategy.generate_signals.return_value = signals
        # Remove generate_signal to ensure generate_signals is used
        del strategy.generate_signal
        
        market_data = {"timestamp": datetime.now(), "prices": {"AAPL": 150.0, "GOOGL": 2800.0}}
        result = self.engine._generate_signals(strategy, market_data)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result, signals)
        strategy.generate_signals.assert_called_once_with(market_data)
    
    def test_generate_signals_none(self):
        """Test signal generation when strategy returns None."""
        strategy = Mock()
        strategy.generate_signal.return_value = None
        
        market_data = {"timestamp": datetime.now(), "prices": {"AAPL": 150.0}}
        signals = self.engine._generate_signals(strategy, market_data)
        
        self.assertEqual(len(signals), 0)


class TestDataIteration(unittest.TestCase):
    """Test data iteration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("100000"),
            symbols=["AAPL", "GOOGL"],
            frequency="1d"
        )
        self.engine = UnifiedBacktestEngine(self.config)
    
    def test_iterate_data(self):
        """Test data iteration with mock data loader."""
        # Create mock data loader
        mock_loader = Mock()
        
        # Create mock data for AAPL
        aapl_data = Mock()
        aapl_data.index = [
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4)
        ]
        aapl_data.loc = {
            datetime(2023, 1, 2): {"open": 150, "high": 152, "low": 149, "close": 151, "volume": 1000000},
            datetime(2023, 1, 3): {"open": 151, "high": 153, "low": 150, "close": 152, "volume": 1100000},
            datetime(2023, 1, 4): {"open": 152, "high": 154, "low": 151, "close": 153, "volume": 1200000}
        }
        
        # Create mock data for GOOGL
        googl_data = Mock()
        googl_data.index = [
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 5)  # Missing Jan 4
        ]
        googl_data.loc = {
            datetime(2023, 1, 2): {"open": 2800, "high": 2820, "low": 2790, "close": 2810, "volume": 500000},
            datetime(2023, 1, 3): {"open": 2810, "high": 2830, "low": 2800, "close": 2820, "volume": 510000},
            datetime(2023, 1, 5): {"open": 2820, "high": 2840, "low": 2810, "close": 2830, "volume": 520000}
        }
        
        # Set up mock loader
        def mock_load(symbol, start, end, frequency):
            if symbol == "AAPL":
                return aapl_data
            elif symbol == "GOOGL":
                return googl_data
            return None
        
        mock_loader.load.side_effect = mock_load
        
        # Iterate through data
        data_points = list(self.engine._iterate_data(mock_loader))
        
        # Should have 4 unique timestamps
        self.assertEqual(len(data_points), 4)
        
        # Check first data point (Jan 2 - both symbols)
        timestamp, market_data = data_points[0]
        self.assertEqual(timestamp, datetime(2023, 1, 2))
        self.assertIn("AAPL", market_data["prices"])
        self.assertIn("GOOGL", market_data["prices"])
        self.assertEqual(market_data["prices"]["AAPL"], 151)
        self.assertEqual(market_data["prices"]["GOOGL"], 2810)
        
        # Check that AAPL data is included in market_data
        self.assertIn("AAPL", market_data)
        self.assertEqual(market_data["AAPL"]["close"], 151)
        
        # Check third data point (Jan 4 - only AAPL)
        timestamp, market_data = data_points[2]
        self.assertEqual(timestamp, datetime(2023, 1, 4))
        self.assertIn("AAPL", market_data["prices"])
        self.assertNotIn("GOOGL", market_data["prices"])


class TestEquityCalculation(unittest.TestCase):
    """Test equity calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
        self.engine = UnifiedBacktestEngine(self.config)
    
    def test_calculate_equity_cash_only(self):
        """Test equity calculation with cash only."""
        market_data = {"prices": {"AAPL": 150.0}}
        
        equity = self.engine._calculate_equity(market_data)
        
        # Should equal initial capital since no positions
        self.assertEqual(equity, Decimal("100000"))
    
    @patch.object(RiskPortfolioContainer, 'get_portfolio_state')
    def test_calculate_equity_with_positions(self, mock_get_portfolio_state):
        """Test equity calculation with positions."""
        # Mock portfolio state
        mock_portfolio_state = Mock()
        mock_risk_metrics = Mock()
        mock_risk_metrics.total_value = Decimal("105000")  # Gained 5%
        mock_portfolio_state.get_risk_metrics.return_value = mock_risk_metrics
        mock_get_portfolio_state.return_value = mock_portfolio_state
        
        market_data = {"prices": {"AAPL": 155.0}}
        
        equity = self.engine._calculate_equity(market_data)
        
        # Should use total value from risk metrics
        self.assertEqual(equity, Decimal("105000"))
        
        # Verify market prices were updated
        mock_portfolio_state.update_market_prices.assert_called_once()
        call_args = mock_portfolio_state.update_market_prices.call_args[0][0]
        self.assertEqual(call_args["AAPL"], Decimal("155.0"))


class TestResultsCalculation(unittest.TestCase):
    """Test backtest results calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
        )
        self.engine = UnifiedBacktestEngine(self.config)
        
        # Create sample equity curve
        self.engine._equity_curve = [
            {"timestamp": datetime(2023, 1, 1), "equity": 100000, "cash": 100000, "positions_value": 0},
            {"timestamp": datetime(2023, 1, 2), "equity": 101000, "cash": 50000, "positions_value": 51000},
            {"timestamp": datetime(2023, 1, 3), "equity": 102000, "cash": 50000, "positions_value": 52000},
            {"timestamp": datetime(2023, 1, 4), "equity": 99000, "cash": 50000, "positions_value": 49000},
            {"timestamp": datetime(2023, 1, 5), "equity": 105000, "cash": 50000, "positions_value": 55000},
        ]
        
        # Create sample daily returns
        self.engine._daily_returns = [
            Decimal("0.01"),    # 1%
            Decimal("0.0099"),  # 0.99%
            Decimal("-0.0294"), # -2.94%
            Decimal("0.0606"),  # 6.06%
        ]
    
    def test_calculate_results_basic_metrics(self):
        """Test calculation of basic metrics."""
        results = self.engine._calculate_results(Decimal("100000"))
        
        # Check basic return metrics
        self.assertEqual(results.initial_capital, Decimal("100000"))
        self.assertEqual(results.final_equity, Decimal("105000"))
        self.assertEqual(results.total_return, Decimal("0.05"))  # 5%
        
        # Check dates
        self.assertEqual(results.start_date, self.config.start_date)
        self.assertEqual(results.end_date, self.config.end_date)
    
    def test_calculate_results_drawdown(self):
        """Test drawdown calculation."""
        results = self.engine._calculate_results(Decimal("100000"))
        
        # Max drawdown should be from 102000 to 99000 = 2.94%
        expected_drawdown = Decimal("3000") / Decimal("102000")
        self.assertAlmostEqual(float(results.max_drawdown), float(expected_drawdown), places=4)
    
    def test_calculate_results_no_trades(self):
        """Test results calculation with no trades."""
        self.engine._trade_history = []
        results = self.engine._calculate_results(Decimal("100000"))
        
        self.assertEqual(results.total_trades, 0)
        self.assertEqual(results.winning_trades, 0)
        self.assertEqual(results.losing_trades, 0)
        self.assertEqual(results.win_rate, Decimal("0"))
        self.assertEqual(results.avg_win, Decimal("0"))
        self.assertEqual(results.avg_loss, Decimal("0"))
        self.assertEqual(results.profit_factor, Decimal("0"))


class TestIntegrationFlow(unittest.TestCase):
    """Test the complete integration flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0"),  # No commission for simplicity
            slippage=Decimal("0")     # No slippage for simplicity
        )
    
    @patch('src.execution.backtest_engine.DefaultExecutionEngine')
    @patch('src.execution.backtest_engine.BacktestBrokerRefactored')
    @patch('src.execution.backtest_engine.RiskPortfolioContainer')
    def test_full_backtest_flow(self, mock_risk_portfolio_class, mock_broker_class, mock_engine_class):
        """Test complete backtest flow from signals to results."""
        # Set up mocks
        mock_risk_portfolio = Mock()
        mock_portfolio_state = Mock()
        mock_risk_portfolio.get_portfolio_state.return_value = mock_portfolio_state
        mock_risk_portfolio_class.return_value = mock_risk_portfolio
        
        # Mock order generation from risk module
        mock_order = Mock(spec=Order)
        mock_order.order_id = "TEST_ORDER_001"
        mock_risk_portfolio.process_signals.return_value = [mock_order]
        
        # Mock portfolio metrics
        mock_metrics = Mock()
        mock_metrics.total_value = Decimal("100000")
        mock_portfolio_state.get_risk_metrics.return_value = mock_metrics
        mock_portfolio_state.get_cash_balance.return_value = Decimal("100000")
        mock_portfolio_state.get_all_positions.return_value = {}
        
        # Create engine
        engine = UnifiedBacktestEngine(self.config)
        
        # Create mock strategy
        strategy = Mock()
        strategy.initialize = Mock()
        signal = Signal(
            signal_id="TEST_SIGNAL_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={"price": 150.0}
        )
        strategy.generate_signals.return_value = [signal]
        
        # Create mock data loader
        mock_loader = Mock()
        mock_data = Mock()
        mock_data.index = [datetime(2023, 1, 2)]
        mock_data.loc = {
            datetime(2023, 1, 2): {"open": 150, "high": 152, "low": 149, "close": 151, "volume": 1000000}
        }
        mock_loader.load.return_value = mock_data
        
        # Run backtest
        results = engine.run(strategy, mock_loader)
        
        # Verify strategy was initialized
        strategy.initialize.assert_called_once()
        
        # Verify signals were processed
        self.assertTrue(mock_risk_portfolio.process_signals.called)
        signals_arg = mock_risk_portfolio.process_signals.call_args[0][0]
        self.assertEqual(len(signals_arg), 1)
        self.assertEqual(signals_arg[0].signal_id, "TEST_SIGNAL_001")
        
        # Verify results
        self.assertIsInstance(results, BacktestResults)
        self.assertEqual(results.initial_capital, Decimal("100000"))
        self.assertEqual(results.final_equity, Decimal("100000"))  # No change since mocked


if __name__ == "__main__":
    unittest.main()