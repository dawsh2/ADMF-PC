"""
Backtest engine for ADMF-PC.

Handles the execution of trading strategies against historical data,
separate from the Coordinator's orchestration responsibilities.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
import logging

from ..core.containers import UniversalScopedContainer
from ..core.events import Event, EventType
from ..data.protocols import DataLoader
from ..strategy.protocols import Strategy, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    data_frequency: str = "1D"  # 1M, 5M, 1H, 1D
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    margin_requirement: float = 1.0  # 1.0 = no leverage
    

@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    

@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    slippage: float
    

class BacktestEngine:
    """
    Handles backtest execution separate from workflow orchestration.
    
    The BacktestEngine:
    - Manages market data flow
    - Executes trades based on signals
    - Tracks portfolio state
    - Calculates performance metrics
    - Handles multi-symbol execution
    """
    
    def __init__(self, 
                 config: BacktestConfig,
                 data_loader: DataLoader,
                 container: Optional[UniversalScopedContainer] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            data_loader: Data loader for market data
            container: Optional container for isolation
        """
        self.config = config
        self.data_loader = data_loader
        self.container = container
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
        # Performance tracking
        self.current_time: Optional[datetime] = None
        self.bar_count = 0
        self.signal_count = 0
        
        # Multi-symbol support
        self.symbol_data: Dict[str, Any] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        
    def run(self, 
            strategies: Dict[str, Strategy],
            risk_container: Optional['RiskContainer'] = None) -> Dict[str, Any]:
        """
        Run backtest with given strategies.
        
        Args:
            strategies: Dict of strategy_id -> strategy instance
            risk_container: Optional risk container for portfolio management
            
        Returns:
            Backtest results including performance metrics
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Load market data for all symbols
        market_data = self._load_market_data()
        
        # Process each bar
        for timestamp, bar_data in market_data:
            self.current_time = timestamp
            self.bar_count += 1
            
            # Update current prices
            self._update_prices(bar_data)
            
            # Update positions with current prices
            self._update_positions()
            
            # Generate signals from strategies
            signals = self._generate_signals(strategies, bar_data)
            
            # Apply risk management if container provided
            if risk_container:
                signals = risk_container.process_signals(signals, self.get_portfolio_state())
            
            # Execute signals
            self._execute_signals(signals)
            
            # Record equity
            self._record_equity()
            
            # Emit progress event
            if self.bar_count % 100 == 0:
                self._emit_progress_event()
        
        # Calculate final metrics
        results = self._calculate_results()
        
        logger.info(f"Backtest complete. Total return: {results['total_return']:.2%}")
        
        return results
    
    def _load_market_data(self) -> List[tuple]:
        """Load and align market data for all symbols."""
        all_data = []
        
        for symbol in self.config.symbols:
            data = self.data_loader.load_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.data_frequency
            )
            self.symbol_data[symbol] = data
        
        # Align data across symbols
        # This is simplified - real implementation would handle missing data
        return self._align_market_data(self.symbol_data)
    
    def _generate_signals(self, 
                         strategies: Dict[str, Strategy],
                         bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from all strategies."""
        all_signals = []
        
        for strategy_id, strategy in strategies.items():
            # Each strategy may generate signals for multiple symbols
            for symbol in self.config.symbols:
                if symbol in bar_data:
                    market_data = {
                        'symbol': symbol,
                        'timestamp': self.current_time,
                        **bar_data[symbol]
                    }
                    
                    signal = strategy.generate_signal(market_data)
                    if signal:
                        signal['strategy_id'] = strategy_id
                        all_signals.append(signal)
                        self.signal_count += 1
        
        return all_signals
    
    def _execute_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Execute trading signals."""
        for signal in signals:
            symbol = signal['symbol']
            direction = signal['direction']
            
            # Calculate position size (simplified)
            position_size = self._calculate_position_size(signal)
            
            if direction == SignalDirection.BUY:
                self._open_long_position(symbol, position_size, signal)
            elif direction == SignalDirection.SELL:
                self._close_long_position(symbol, signal)
    
    def _open_long_position(self, symbol: str, size: float, signal: Dict[str, Any]) -> None:
        """Open a long position."""
        if symbol in self.positions:
            return  # Already have position
        
        price = self.symbol_data[symbol]['close']
        cost = price * size * (1 + self.config.commission + self.config.slippage)
        
        if cost > self.cash:
            return  # Insufficient funds
        
        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=size,
            entry_price=price * (1 + self.config.slippage),
            entry_time=self.current_time,
            current_price=price
        )
        
        self.cash -= cost
        
        # Emit trade event
        self._emit_trade_event('open', symbol, size, price)
    
    def _close_long_position(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Close a long position."""
        if symbol not in self.positions:
            return  # No position to close
        
        position = self.positions[symbol]
        price = self.symbol_data[symbol]['close'] * (1 - self.config.slippage)
        
        # Calculate P&L
        gross_pnl = (price - position.entry_price) * position.quantity
        commission = price * position.quantity * self.config.commission
        net_pnl = gross_pnl - commission
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=net_pnl,
            commission=commission,
            slippage=self.config.slippage * price * position.quantity
        )
        self.trades.append(trade)
        
        # Update cash
        self.cash += price * position.quantity - commission
        
        # Remove position
        del self.positions[symbol]
        
        # Emit trade event
        self._emit_trade_event('close', symbol, position.quantity, price)
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size based on signal and risk rules."""
        # Simplified - use fixed percentage of capital
        symbol = signal['symbol']
        price = self.symbol_data[symbol]['close']
        
        # 2% of capital per position
        position_value = self.cash * 0.02
        position_size = position_value / price
        
        return position_size
    
    def _update_positions(self) -> None:
        """Update position values with current prices."""
        for symbol, position in self.positions.items():
            if symbol in self.symbol_data:
                current_price = self.symbol_data[symbol]['close']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def _record_equity(self) -> None:
        """Record current portfolio equity."""
        positions_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        total_equity = self.cash + positions_value
        self.equity_curve.append(total_equity)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not self.equity_curve:
            return {}
        
        initial_equity = self.config.initial_capital
        final_equity = self.equity_curve[-1]
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        # Calculate metrics
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Sharpe ratio (simplified - assumes daily returns)
        if returns:
            import numpy as np
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'initial_capital': initial_equity,
            'final_value': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.pnl > 0),
            'losing_trades': sum(1 for t in self.trades if t.pnl < 0),
            'total_commission': sum(t.commission for t in self.trades),
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'returns': returns,
            'bars_processed': self.bar_count,
            'signals_generated': self.signal_count
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.equity_curve:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'cash': self.cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            },
            'total_equity': self.cash + sum(
                pos.current_price * pos.quantity 
                for pos in self.positions.values()
            )
        }
    
    def _align_market_data(self, symbol_data: Dict[str, Any]) -> List[tuple]:
        """Align market data across symbols."""
        # Simplified implementation
        aligned_data = []
        
        # Get all timestamps
        all_timestamps = set()
        for symbol, data in symbol_data.items():
            all_timestamps.update(data.keys())
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Create aligned data
        for timestamp in sorted_timestamps:
            bar_data = {}
            for symbol, data in symbol_data.items():
                if timestamp in data:
                    bar_data[symbol] = data[timestamp]
            
            if bar_data:  # Only include if we have data for at least one symbol
                aligned_data.append((timestamp, bar_data))
        
        return aligned_data
    
    def _update_prices(self, bar_data: Dict[str, Any]) -> None:
        """Update current prices for all symbols."""
        for symbol, data in bar_data.items():
            if 'close' in data:
                if symbol not in self.symbol_data:
                    self.symbol_data[symbol] = {}
                self.symbol_data[symbol]['close'] = data['close']
    
    def _emit_progress_event(self) -> None:
        """Emit backtest progress event."""
        if self.container and self.container.event_bus:
            event = Event(
                event_type=EventType.INFO,
                payload={
                    'type': 'backtest.progress',
                    'bars_processed': self.bar_count,
                    'current_time': self.current_time.isoformat() if self.current_time else None,
                    'equity': self.equity_curve[-1] if self.equity_curve else 0
                },
                source_id="backtest_engine",
                container_id=self.container.container_id
            )
            self.container.event_bus.publish(event)
    
    def _emit_trade_event(self, action: str, symbol: str, quantity: float, price: float) -> None:
        """Emit trade execution event."""
        if self.container and self.container.event_bus:
            event = Event(
                event_type=EventType.INFO,
                payload={
                    'type': f'trade.{action}',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': self.current_time.isoformat() if self.current_time else None
                },
                source_id="backtest_engine",
                container_id=self.container.container_id
            )
            self.container.event_bus.publish(event)