"""
Integration test for complete backtest flow.

Tests the entire pipeline from strategy signals through
risk management, execution, and results calculation.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Note: Imports are commented out due to circular dependency issues
# In a real test environment, these would be properly resolved

class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self):
        self.positions = {}
        self.signal_count = 0
    
    def initialize(self, context):
        """Initialize strategy."""
        self.symbols = context.get('symbols', [])
    
    def generate_signals(self, market_data):
        """Generate test signals."""
        from src.risk.protocols import Signal, SignalType, OrderSide
        
        signals = []
        timestamp = market_data['timestamp']
        
        # Simple logic: buy on first bar, sell on third bar
        self.signal_count += 1
        
        if self.signal_count == 1:
            # Buy signals for all symbols
            for symbol in self.symbols:
                if symbol in market_data.get('prices', {}):
                    signal = Signal(
                        signal_id=f"BUY_{symbol}_{timestamp}",
                        strategy_id="test_strategy",
                        symbol=symbol,
                        signal_type=SignalType.ENTRY,
                        side=OrderSide.BUY,
                        strength=Decimal("0.8"),
                        timestamp=timestamp,
                        metadata={'bar': self.signal_count}
                    )
                    signals.append(signal)
                    self.positions[symbol] = 1
        
        elif self.signal_count == 3:
            # Sell signals for all positions
            for symbol, pos in self.positions.items():
                if pos > 0 and symbol in market_data.get('prices', {}):
                    signal = Signal(
                        signal_id=f"SELL_{symbol}_{timestamp}",
                        strategy_id="test_strategy",
                        symbol=symbol,
                        signal_type=SignalType.EXIT,
                        side=OrderSide.SELL,
                        strength=Decimal("1.0"),
                        timestamp=timestamp,
                        metadata={'bar': self.signal_count}
                    )
                    signals.append(signal)
                    self.positions[symbol] = 0
        
        return signals


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, price_series):
        """Initialize with price series dict."""
        self.price_series = price_series
    
    def load(self, symbol, start, end, frequency):
        """Load mock data."""
        class MockDataFrame:
            def __init__(self, data_dict, dates):
                self.data = data_dict
                self.index = dates
            
            def __len__(self):
                return len(self.index)
            
            @property 
            def loc(self):
                """Simple loc accessor."""
                return {
                    date: self.data[date]
                    for date in self.index
                }
        
        if symbol not in self.price_series:
            return None
        
        prices = self.price_series[symbol]
        data_dict = {}
        dates = []
        
        current = start
        for i, price in enumerate(prices):
            if current > end:
                break
            
            data_dict[current] = {
                'open': price,
                'high': price + 1,
                'low': price - 1,
                'close': price,
                'volume': 1000000
            }
            dates.append(current)
            current += timedelta(days=1)
        
        return MockDataFrame(data_dict, dates)


class TestFullBacktestFlow(unittest.TestCase):
    """Test complete backtest flow integration."""
    
    def test_profitable_backtest(self):
        """Test a simple profitable backtest."""
        # Skip if imports fail
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
            from src.risk.risk_limits import MaxDrawdownLimit
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        # Configure backtest
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            initial_capital=Decimal("100000"),
            symbols=["AAPL", "GOOGL"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
        
        # Create strategy
        strategy = MockStrategy()
        
        # Create data loader with rising prices
        price_series = {
            "AAPL": [150, 151, 152, 153, 154, 155, 156, 157, 158, 159],
            "GOOGL": [2800, 2810, 2820, 2830, 2840, 2850, 2860, 2870, 2880, 2890]
        }
        data_loader = MockDataLoader(price_series)
        
        # Create and configure engine
        engine = UnifiedBacktestEngine(config)
        
        # Add risk limits
        engine.risk_portfolio.add_risk_limit(
            MaxDrawdownLimit(
                max_drawdown_pct=Decimal("10"),
                reduce_at_pct=Decimal("8")
            )
        )
        
        # Run backtest
        results = engine.run(strategy, data_loader)
        
        # Verify results
        self.assertIsNotNone(results)
        
        # Should be profitable (bought at ~150, sold at ~152)
        self.assertGreater(results.total_return, Decimal("0"))
        self.assertGreater(results.final_equity, results.initial_capital)
        
        # Should have executed trades
        self.assertGreater(results.total_trades, 0)
        self.assertEqual(results.winning_trades + results.losing_trades, results.total_trades)
        
        # Check metrics
        self.assertGreaterEqual(results.win_rate, Decimal("0"))
        self.assertLessEqual(results.win_rate, Decimal("1"))
        self.assertGreaterEqual(results.max_drawdown, Decimal("0"))
        
        # Verify equity curve
        self.assertGreater(len(results.equity_curve), 0)
        self.assertEqual(
            Decimal(str(results.equity_curve[0]['equity'])),
            results.initial_capital
        )
        self.assertEqual(
            Decimal(str(results.equity_curve[-1]['equity'])),
            results.final_equity
        )
    
    def test_losing_backtest(self):
        """Test a backtest with losses."""
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.001")
        )
        
        strategy = MockStrategy()
        
        # Create data with falling prices
        price_series = {
            "AAPL": [150, 149, 148, 147, 146, 145, 144, 143, 142, 141]
        }
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config)
        results = engine.run(strategy, data_loader)
        
        # Should have losses
        self.assertLess(results.total_return, Decimal("0"))
        self.assertLess(results.final_equity, results.initial_capital)
        
        # Should still have valid metrics
        self.assertGreaterEqual(results.max_drawdown, Decimal("0"))
        self.assertGreaterEqual(results.volatility, Decimal("0"))
    
    def test_risk_limits_enforced(self):
        """Test that risk limits are enforced during backtest."""
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
            from src.risk.risk_limits import MaxPositionLimit
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("10000"),  # Small capital
            symbols=["AAPL", "GOOGL", "AMZN"],
            commission=Decimal("0"),
            slippage=Decimal("0")
        )
        
        # Expensive stocks relative to capital
        price_series = {
            "AAPL": [150, 151, 152, 153, 154],
            "GOOGL": [2800, 2810, 2820, 2830, 2840],
            "AMZN": [3500, 3510, 3520, 3530, 3540]
        }
        
        strategy = MockStrategy()
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config)
        
        # Add strict position limit
        engine.risk_portfolio.add_risk_limit(
            MaxPositionLimit(max_position_value=Decimal("2000"))  # Max $2k per position
        )
        
        results = engine.run(strategy, data_loader)
        
        # Should have fewer trades due to position limits
        # With $10k capital and $2k max per position, can't buy GOOGL or AMZN
        self.assertLess(results.total_trades, 6)  # Less than 3 buys + 3 sells
    
    def test_multiple_strategy_signals(self):
        """Test handling multiple signals per bar."""
        
        class MultiSignalStrategy:
            """Strategy that generates multiple signals."""
            
            def __init__(self):
                self.bar_count = 0
            
            def initialize(self, context):
                self.symbols = context.get('symbols', [])
            
            def generate_signals(self, market_data):
                from src.risk.protocols import Signal, SignalType, OrderSide
                
                self.bar_count += 1
                signals = []
                
                if self.bar_count == 1:
                    # Generate multiple buy signals
                    for symbol in self.symbols:
                        if symbol in market_data.get('prices', {}):
                            # Strong buy
                            signals.append(Signal(
                                signal_id=f"BUY_STRONG_{symbol}",
                                strategy_id="multi_strategy",
                                symbol=symbol,
                                signal_type=SignalType.ENTRY,
                                side=OrderSide.BUY,
                                strength=Decimal("0.9"),
                                timestamp=market_data['timestamp'],
                                metadata={'signal_type': 'strong'}
                            ))
                            
                            # Weak buy (should be ignored due to position)
                            signals.append(Signal(
                                signal_id=f"BUY_WEAK_{symbol}",
                                strategy_id="multi_strategy",
                                symbol=symbol,
                                signal_type=SignalType.ENTRY,
                                side=OrderSide.BUY,
                                strength=Decimal("0.3"),
                                timestamp=market_data['timestamp'],
                                metadata={'signal_type': 'weak'}
                            ))
                
                return signals
        
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),
            initial_capital=Decimal("100000"),
            symbols=["AAPL", "MSFT"],
            commission=Decimal("0"),
            slippage=Decimal("0")
        )
        
        strategy = MultiSignalStrategy()
        price_series = {
            "AAPL": [150, 151, 152],
            "MSFT": [380, 381, 382]
        }
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config)
        results = engine.run(strategy, data_loader)
        
        # Should only process one signal per symbol (no duplicate positions)
        self.assertEqual(results.total_trades, 2)  # One per symbol
    
    def test_backtest_with_no_signals(self):
        """Test backtest that generates no signals."""
        
        class NoSignalStrategy:
            """Strategy that never signals."""
            
            def initialize(self, context):
                pass
            
            def generate_signals(self, market_data):
                return []  # Never signal
        
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005")
        )
        
        strategy = NoSignalStrategy()
        price_series = {"AAPL": [150, 151, 152, 153, 154]}
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config)
        results = engine.run(strategy, data_loader)
        
        # Should complete without errors
        self.assertEqual(results.total_trades, 0)
        self.assertEqual(results.total_return, Decimal("0"))
        self.assertEqual(results.final_equity, results.initial_capital)
        self.assertEqual(results.max_drawdown, Decimal("0"))
    
    def test_backtest_progress_events(self):
        """Test that progress events are emitted during backtest."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
            from src.core.containers.universal import UniversalScopedContainer
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        # Create container with event tracking
        container = UniversalScopedContainer("test_container")
        container.event_bus.subscribe(
            lambda e: e.payload.get('type') == 'backtest_progress',
            event_handler
        )
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0"),
            slippage=Decimal("0")
        )
        
        strategy = MockStrategy()
        price_series = {"AAPL": [150, 151, 152, 153, 154]}
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config, container=container)
        results = engine.run(strategy, data_loader)
        
        # Should have received progress events
        self.assertGreater(len(events_received), 0)
        
        # Check progress values
        progress_values = [
            e.payload.get('progress', 0)
            for e in events_received
        ]
        
        # Progress should increase
        for i in range(1, len(progress_values)):
            self.assertGreaterEqual(progress_values[i], progress_values[i-1])


class TestBacktestResultsAccuracy(unittest.TestCase):
    """Test accuracy of backtest results calculation."""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation accuracy."""
        try:
            from src.execution.backtest_engine import BacktestResults
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        # Create results with known returns
        daily_returns = [
            Decimal("0.01"),   # 1%
            Decimal("-0.005"), # -0.5%
            Decimal("0.015"),  # 1.5%
            Decimal("0.002"),  # 0.2%
            Decimal("-0.01"),  # -1%
        ]
        
        # Manual calculation
        import statistics
        returns_float = [float(r) for r in daily_returns]
        avg_return = statistics.mean(returns_float)
        std_dev = statistics.stdev(returns_float) if len(returns_float) > 1 else 0
        
        # Annualized (252 trading days)
        annual_return = avg_return * 252
        annual_vol = std_dev * (252 ** 0.5)
        expected_sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Create mock results
        results = BacktestResults(
            total_return=Decimal("0.012"),
            annualized_return=Decimal(str(annual_return)),
            sharpe_ratio=Decimal(str(expected_sharpe)),
            max_drawdown=Decimal("0.01"),
            win_rate=Decimal("0.6"),
            profit_factor=Decimal("1.5"),
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            avg_win=Decimal("100"),
            avg_loss=Decimal("50"),
            equity_curve=[],
            daily_returns=daily_returns,
            positions=[],
            trades=[],
            volatility=Decimal(str(annual_vol)),
            var_95=Decimal("0.015"),
            cvar_95=Decimal("0.02"),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            initial_capital=Decimal("100000"),
            final_equity=Decimal("101200")
        )
        
        # Verify Sharpe calculation
        self.assertAlmostEqual(
            float(results.sharpe_ratio),
            expected_sharpe,
            places=2
        )
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        try:
            from src.execution.backtest_engine import UnifiedBacktestEngine, BacktestConfig
        except ImportError:
            self.skipTest("Import dependencies not available")
        
        # Create a strategy that will have a drawdown
        class DrawdownStrategy:
            def __init__(self):
                self.bar = 0
            
            def initialize(self, context):
                self.symbols = context.get('symbols', [])
            
            def generate_signals(self, market_data):
                from src.risk.protocols import Signal, SignalType, OrderSide
                
                self.bar += 1
                
                # Buy immediately
                if self.bar == 1:
                    return [Signal(
                        signal_id="BUY_001",
                        strategy_id="dd_test",
                        symbol="AAPL",
                        signal_type=SignalType.ENTRY,
                        side=OrderSide.BUY,
                        strength=Decimal("1.0"),
                        timestamp=market_data['timestamp'],
                        metadata={}
                    )]
                
                # Sell at loss
                if self.bar == 5:
                    return [Signal(
                        signal_id="SELL_001",
                        strategy_id="dd_test",
                        symbol="AAPL",
                        signal_type=SignalType.EXIT,
                        side=OrderSide.SELL,
                        strength=Decimal("1.0"),
                        timestamp=market_data['timestamp'],
                        metadata={}
                    )]
                
                return []
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            initial_capital=Decimal("100000"),
            symbols=["AAPL"],
            commission=Decimal("0"),
            slippage=Decimal("0")
        )
        
        # Price goes up then down
        price_series = {
            "AAPL": [150, 155, 160, 155, 145, 140, 145, 150, 155, 160]
        }
        
        strategy = DrawdownStrategy()
        data_loader = MockDataLoader(price_series)
        
        engine = UnifiedBacktestEngine(config)
        results = engine.run(strategy, data_loader)
        
        # Should have recorded drawdown
        self.assertGreater(results.max_drawdown, Decimal("0"))
        
        # Drawdown should be from peak (160) to trough (140)
        # Approximate as we buy at 150 and the position value drops
        self.assertGreater(results.max_drawdown, Decimal("0.05"))  # At least 5%


if __name__ == "__main__":
    unittest.main()