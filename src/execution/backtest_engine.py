"""
Unified backtest engine that lives within the execution module.

This module combines the best of both backtest and execution modules,
delegating execution to the execution engine and position tracking to
the risk module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
import logging

from ..core.events.types import Event, EventType
from ..core.containers.universal import UniversalScopedContainer
from ..data.protocols import DataLoader
from ..strategy.protocols import Strategy
from ..risk.protocols import Signal
from ..risk.risk_portfolio import RiskPortfolioContainer
from .execution_engine import DefaultExecutionEngine
from .backtest_broker_refactored import BacktestBrokerRefactored
from .market_simulation import MarketSimulator
from .order_manager import OrderManager


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    symbols: List[str]
    frequency: str = "1d"
    commission: Decimal = Decimal("0.001")  # 0.1%
    slippage: Decimal = Decimal("0.0005")   # 0.05%
    
    # Additional configuration
    enable_shorting: bool = True
    use_adjusted_close: bool = True
    rebalance_frequency: Optional[str] = None
    benchmark_symbol: Optional[str] = None


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    # Performance metrics
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    
    # Time series data
    equity_curve: List[Dict[str, Any]]
    daily_returns: List[Decimal]
    positions: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    
    # Risk metrics
    volatility: Decimal
    var_95: Decimal
    cvar_95: Decimal
    
    # Additional info
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_equity: Decimal


class UnifiedBacktestEngine:
    """
    Unified backtesting engine that orchestrates backtest execution.
    
    This engine:
    - Uses ExecutionEngine for order processing
    - Delegates position tracking to Risk module
    - Provides comprehensive performance analytics
    - Maintains single source of truth for state
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        container: Optional[UniversalScopedContainer] = None
    ):
        """
        Initialize the unified backtest engine.
        
        Args:
            config: Backtest configuration
            container: Optional container for isolation
        """
        self.config = config
        self.container = container
        
        # Initialize components
        self._initialize_components()
        
        # Results tracking
        self._equity_curve: List[Dict[str, Any]] = []
        self._daily_returns: List[Decimal] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._current_date: Optional[datetime] = None
        
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        # Risk & Portfolio container
        self.risk_portfolio = RiskPortfolioContainer(
            name="backtest_portfolio",
            initial_capital=self.config.initial_capital
        )
        
        # Backtest broker (uses risk module's portfolio state)
        self.broker = BacktestBrokerRefactored(
            portfolio_state=self.risk_portfolio.get_portfolio_state(),
            initial_cash=self.config.initial_capital
        )
        
        # Market simulator
        from .market_simulation import FixedSlippageModel, PercentCommissionModel
        slippage_model = FixedSlippageModel(slippage_percent=float(self.config.slippage))
        commission_model = PercentCommissionModel(
            commission_percent=float(self.config.commission),
            min_commission=0.0  # No minimum commission for testing
        )
        
        self.market_simulator = MarketSimulator(
            slippage_model=slippage_model,
            commission_model=commission_model
        )
        
        # Order manager
        self.order_manager = OrderManager()
        
        # Execution engine
        self.execution_engine = DefaultExecutionEngine(
            broker=self.broker,
            order_manager=self.order_manager,
            market_simulator=self.market_simulator
        )
        
        # Set execution context for backtesting
        self.execution_engine._mode = "backtest"
        
    def run(
        self,
        strategy: Strategy,
        data_loader: DataLoader
    ) -> BacktestResults:
        """
        Run backtest with the given strategy.
        
        Args:
            strategy: Trading strategy to test
            data_loader: Data source for market data
            
        Returns:
            BacktestResults with comprehensive metrics
        """
        logger.info(
            f"Starting backtest from {self.config.start_date} to {self.config.end_date}"
        )
        
        # Initialize strategy
        if hasattr(strategy, 'initialize'):
            strategy.initialize({'symbols': self.config.symbols})
        
        # Track starting equity
        starting_equity = self.config.initial_capital
        previous_equity = starting_equity
        
        # Main backtest loop
        for timestamp, market_data in self._iterate_data(data_loader):
            self._current_date = timestamp
            
            # Update market data in execution engine
            self.execution_engine.process_event(
                Event(
                    event_type=EventType.BAR,
                    payload=market_data,
                    timestamp=timestamp
                )
            )
            
            # Generate signals from strategy
            signals = self._generate_signals(strategy, market_data)
            
            # Process signals through risk module
            if signals:
                orders = self.risk_portfolio.process_signals(signals, market_data)
                
                # Submit orders through execution engine
                for order in orders:
                    self.execution_engine.process_event(
                        Event(
                            event_type=EventType.ORDER,
                            payload={
                                'order': order,
                                'source': 'backtest_engine'
                            },
                            timestamp=timestamp
                        )
                    )
            
            # Update portfolio metrics
            current_equity = self._calculate_equity(market_data)
            daily_return = (current_equity - previous_equity) / previous_equity
            self._daily_returns.append(daily_return)
            previous_equity = current_equity
            
            # Record equity curve point
            self._equity_curve.append({
                'timestamp': timestamp,
                'equity': float(current_equity),
                'cash': float(self.risk_portfolio.get_portfolio_state().get_cash_balance()),
                'positions_value': float(current_equity - self.risk_portfolio.get_portfolio_state().get_cash_balance())
            })
            
            # Emit progress event
            if self.container:
                progress = (timestamp - self.config.start_date) / (self.config.end_date - self.config.start_date)
                self.container.event_bus.publish(
                    Event(
                        event_type=EventType.INFO,
                        payload={
                            'type': 'backtest_progress',
                            'progress': float(progress),
                            'current_date': timestamp,
                            'equity': float(current_equity)
                        }
                    )
                )
        
        # Calculate final results
        results = self._calculate_results(starting_equity)
        
        logger.info(
            f"Backtest complete. Total return: {results.total_return:.2%}, "
            f"Sharpe ratio: {results.sharpe_ratio:.2f}"
        )
        
        return results
    
    def _iterate_data(self, data_loader: DataLoader):
        """Iterate through market data."""
        # Load data for all symbols
        all_data = {}
        for symbol in self.config.symbols:
            data = data_loader.load(
                symbol=symbol,
                start=self.config.start_date,
                end=self.config.end_date,
                frequency=self.config.frequency
            )
            if data is not None and len(data) > 0:
                all_data[symbol] = data
        
        if not all_data:
            raise ValueError("No data loaded for any symbols")
        
        # Get all unique timestamps
        all_timestamps = set()
        for symbol_data in all_data.values():
            all_timestamps.update(symbol_data.index)
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Iterate through timestamps
        for timestamp in sorted_timestamps:
            market_data = {
                'timestamp': timestamp,
                'prices': {}
            }
            
            # Collect data for all symbols at this timestamp
            for symbol, data in all_data.items():
                if timestamp in data.index:
                    row = data.loc[timestamp]
                    market_data['prices'][symbol] = float(row['close'])
                    market_data[symbol] = {
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row.get('volume', 0))
                    }
            
            if market_data['prices']:  # Only yield if we have price data
                yield timestamp, market_data
    
    def _generate_signals(
        self,
        strategy: Strategy,
        market_data: Dict[str, Any]
    ) -> List[Signal]:
        """Generate signals from strategy."""
        signals = []
        
        # Call strategy's signal generation
        if hasattr(strategy, 'generate_signals'):
            # Multi-signal strategy
            strategy_signals = strategy.generate_signals(market_data)
            if strategy_signals:
                signals.extend(strategy_signals)
        elif hasattr(strategy, 'generate_signal'):
            # Single-signal strategy
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_equity(self, market_data: Dict[str, Any]) -> Decimal:
        """Calculate current portfolio equity."""
        portfolio_state = self.risk_portfolio.get_portfolio_state()
        
        # Update market prices
        prices = {
            symbol: Decimal(str(price))
            for symbol, price in market_data.get('prices', {}).items()
        }
        portfolio_state.update_market_prices(prices)
        
        # Get total portfolio value
        metrics = portfolio_state.get_risk_metrics()
        return metrics.total_value
    
    def _calculate_results(self, starting_equity: Decimal) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        if not self._equity_curve:
            raise ValueError("No equity curve data")
        
        final_equity = Decimal(str(self._equity_curve[-1]['equity']))
        
        # Basic returns
        total_return = (final_equity - starting_equity) / starting_equity
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else Decimal(0)
        
        # Risk metrics
        returns_array = [float(r) for r in self._daily_returns if r != 0]
        if returns_array:
            import statistics
            volatility = Decimal(str(statistics.stdev(returns_array))) if len(returns_array) > 1 else Decimal(0)
            annual_volatility = volatility * Decimal(str(252 ** 0.5))
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / annual_volatility if annual_volatility > 0 else Decimal(0)
        else:
            volatility = Decimal(0)
            annual_volatility = Decimal(0)
            sharpe_ratio = Decimal(0)
        
        # Drawdown calculation
        peak = starting_equity
        max_drawdown = Decimal(0)
        for point in self._equity_curve:
            equity = Decimal(str(point['equity']))
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else Decimal(0)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trade statistics
        trades = self._get_trade_history()
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_trades = len(trades)
        win_rate = Decimal(len(winning_trades)) / Decimal(total_trades) if total_trades > 0 else Decimal(0)
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else Decimal(0)
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else Decimal(0)
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else Decimal(0)
        
        # VaR and CVaR (simplified)
        if returns_array:
            sorted_returns = sorted(returns_array)
            var_index = int(len(sorted_returns) * 0.05)
            var_95 = Decimal(str(-sorted_returns[var_index])) if var_index < len(sorted_returns) else Decimal(0)
            cvar_95 = Decimal(str(-sum(sorted_returns[:var_index]) / var_index)) if var_index > 0 else Decimal(0)
        else:
            var_95 = Decimal(0)
            cvar_95 = Decimal(0)
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=self._equity_curve,
            daily_returns=self._daily_returns,
            positions=self._get_position_history(),
            trades=trades,
            volatility=annual_volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=starting_equity,
            final_equity=final_equity
        )
    
    def _get_trade_history(self) -> List[Dict[str, Any]]:
        """Get completed trades from broker."""
        trades = []
        
        # Get fills from broker
        if hasattr(self.broker, 'get_fills'):
            fills = self.broker.get_fills()
            
            # Group fills by symbol to identify trades
            position_tracker = {}
            
            for fill in fills:
                symbol = fill.symbol
                quantity = fill.quantity if fill.side == 'buy' else -fill.quantity
                
                if symbol not in position_tracker:
                    position_tracker[symbol] = {
                        'quantity': Decimal(0),
                        'cost_basis': Decimal(0),
                        'entry_fill': None
                    }
                
                pos = position_tracker[symbol]
                
                # Check if this closes a position
                if pos['quantity'] != 0 and (pos['quantity'] > 0) != (quantity > 0):
                    # Closing trade
                    trade_quantity = min(abs(pos['quantity']), abs(quantity))
                    entry_price = pos['cost_basis'] / abs(pos['quantity'])
                    exit_price = fill.price
                    
                    if pos['quantity'] > 0:  # Long trade
                        pnl = (exit_price - entry_price) * trade_quantity
                    else:  # Short trade
                        pnl = (entry_price - exit_price) * trade_quantity
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_fill'].timestamp,
                        'exit_time': fill.timestamp,
                        'quantity': float(trade_quantity),
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'pnl': float(pnl),
                        'return': float((exit_price - entry_price) / entry_price) if pos['quantity'] > 0 else float((entry_price - exit_price) / entry_price)
                    })
                    
                    # Update position
                    pos['quantity'] += quantity
                    if abs(pos['quantity']) < 0.0001:  # Position closed
                        pos['quantity'] = Decimal(0)
                        pos['cost_basis'] = Decimal(0)
                        pos['entry_fill'] = None
                else:
                    # Opening or adding to position
                    pos['quantity'] += quantity
                    pos['cost_basis'] += abs(quantity) * fill.price
                    if pos['entry_fill'] is None:
                        pos['entry_fill'] = fill
        
        return trades
    
    def _get_position_history(self) -> List[Dict[str, Any]]:
        """Get position history from portfolio."""
        positions = []
        
        # Get current positions from portfolio
        portfolio_state = self.risk_portfolio.get_portfolio_state()
        for symbol, position in portfolio_state.get_all_positions().items():
            if position.quantity != 0:
                positions.append({
                    'symbol': symbol,
                    'quantity': float(position.quantity),
                    'market_value': float(position.market_value),
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'cost_basis': float(position.cost_basis)
                })
        
        return positions