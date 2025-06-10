"""
Event-based metrics tracking for containers.

This module provides a unified approach where event tracing IS the metrics system.
Uses smart retention policies to keep memory usage minimal while calculating
all necessary performance metrics from the event stream.
"""

from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import math
import logging

from ..types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """
    Calculate performance metrics without storing full history.
    
    Uses Welford's algorithm for numerically stable calculation of
    mean and variance without storing all data points.
    
    Note: In minimal mode (no equity curve), max_drawdown will be 0
    as we can't track it without historical values.
    """
    
    # Performance tracking
    total_return: float = 0.0
    n_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Drawdown tracking (requires equity curve)
    max_drawdown: float = 0.0  # Will be 0 in minimal mode
    current_drawdown: float = 0.0
    peak_value: float = 0.0
    
    # Return statistics (Welford's algorithm)
    n_returns: int = 0
    mean_return: float = 0.0
    m2: float = 0.0  # Sum of squares of differences
    
    # P&L tracking
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Portfolio state
    initial_capital: float = 100000.0
    current_value: float = 100000.0
    
    # Time tracking
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Configuration
    annualization_factor: float = 252.0
    min_periods: int = 20
    
    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update metrics with new portfolio value."""
        if self.start_time is None:
            self.start_time = timestamp or datetime.now()
            self.peak_value = value
        
        self.last_update = timestamp or datetime.now()
        
        # Calculate return
        if self.current_value > 0:
            returns = (value - self.current_value) / self.current_value
            
            # Update return statistics (Welford's algorithm)
            self.n_returns += 1
            delta = returns - self.mean_return
            self.mean_return += delta / self.n_returns
            delta2 = returns - self.mean_return
            self.m2 += delta * delta2
        
        # Update portfolio value
        self.current_value = value
        
        # Update total return
        if self.initial_capital > 0:
            self.total_return = (value - self.initial_capital) / self.initial_capital
        
        # Update drawdown
        if value > self.peak_value:
            self.peak_value = value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def update_from_trade(self, entry_price: float, exit_price: float, 
                         quantity: float, direction: str) -> None:
        """Update metrics from a completed trade."""
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity
        
        self.n_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return current metrics."""
        # Calculate variance and Sharpe
        variance = self.m2 / (self.n_returns - 1) if self.n_returns > 1 else 0.0
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (self.mean_return / std_dev * math.sqrt(self.annualization_factor)
                  if std_dev > 0 and self.n_returns >= self.min_periods else 0.0)
        
        # Calculate win rate and profit factor
        win_rate = self.winning_trades / self.n_trades if self.n_trades > 0 else 0.0
        profit_factor = (self.gross_profit / self.gross_loss 
                        if self.gross_loss > 0 else float('inf'))
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': self.n_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'current_value': self.current_value,
            'peak_value': self.peak_value,
        }


class MetricsEventTracer:
    """
    Event tracer optimized for metrics calculation.
    
    Processes events to calculate metrics then discards them based on
    retention policy. This provides a unified system where event tracing
    IS the metrics system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics event tracer.
        
        Args:
            config: Configuration dict with:
                - initial_capital: Starting capital
                - retention_policy: 'trade_complete', 'sliding_window', 'minimal'
                - max_events: Maximum events to retain in memory
                - store_equity_curve: Whether to keep equity snapshots
                - snapshot_interval: How often to snapshot equity
                - objective_function: Dict with function name and params for optimization
                - custom_metrics: List of custom metric calculators
        """
        # Configuration
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.retention_policy = config.get('retention_policy', 'trade_complete')
        self.max_events = config.get('max_events', 1000)
        
        # Objective function configuration
        self.objective_config = config.get('objective_function', {
            'name': 'sharpe_ratio',
            'params': {}
        })
        self.custom_metrics = config.get('custom_metrics', [])
        
        # Results collection config
        collection_config = config.get('collection', {})
        self.store_equity_curve = collection_config.get('store_equity_curve', False)
        self.snapshot_interval = collection_config.get('snapshot_interval', 100)
        self.store_trades = collection_config.get('store_trades', True)
        
        # Initialize metrics
        self.metrics = StreamingMetrics(
            initial_capital=self.initial_capital,
            current_value=self.initial_capital
        )
        
        # Event storage based on retention policy
        if self.retention_policy == 'sliding_window':
            self.events = deque(maxlen=self.max_events)
        else:
            self.events = deque()  # Unlimited but we'll manage it
        
        # Trade tracking
        self.active_trades: Dict[str, List[Event]] = {}  # order_id -> events
        self.completed_trades: List[Dict[str, Any]] = []
        
        # Position tracking for trade matching
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        
        # Equity curve tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.update_counter = 0
        self.last_snapshot = 0
        
        logger.info(f"MetricsEventTracer initialized: "
                   f"retention={self.retention_policy}, "
                   f"max_events={self.max_events}, "
                   f"equity_curve={self.store_equity_curve}")
    
    def trace_event(self, event: Event) -> None:
        """
        Process an event for metrics calculation.
        
        Events are processed immediately for metrics then retained
        or discarded based on retention policy.
        """
        # Store event based on policy
        if self.retention_policy == 'sliding_window':
            self.events.append(event)
        elif self.retention_policy == 'trade_complete':
            # Only store if part of active trade
            self._store_trade_event(event)
        
        # Process event for metrics
        if event.event_type == EventType.ORDER_REQUEST:
            self._process_order_request(event)
        elif event.event_type == EventType.ORDER:
            self._process_order(event)
        elif event.event_type == EventType.FILL:
            self._process_fill(event)
        elif event.event_type == EventType.PORTFOLIO_UPDATE:
            self._process_portfolio_update(event)
        elif event.event_type == EventType.POSITION_UPDATE:
            self._process_position_update(event)
    
    def _store_trade_event(self, event: Event) -> None:
        """Store event if it's part of an active trade."""
        # Use correlation_id if available, fallback to order_id extraction
        trade_id = event.correlation_id
        
        if not trade_id:
            # Fallback to extracting order_id from payload
            if event.event_type in (EventType.ORDER_REQUEST, EventType.ORDER):
                trade_id = event.payload.get('order', {}).get('id')
            elif event.event_type == EventType.FILL:
                trade_id = event.payload.get('order_id')
        
        if trade_id:
            if trade_id not in self.active_trades:
                self.active_trades[trade_id] = []
            self.active_trades[trade_id].append(event)
    
    def _process_order_request(self, event: Event) -> None:
        """Track order request."""
        # Use correlation_id if available
        trade_id = event.correlation_id or event.payload.get('order', {}).get('id')
        
        if trade_id and self.retention_policy == 'trade_complete':
            self.active_trades[trade_id] = [event]
    
    def _process_order(self, event: Event) -> None:
        """Track validated order."""
        # Order has been risk-validated
        pass
    
    def _process_fill(self, event: Event) -> None:
        """Process fill event to update positions and check for completed trades."""
        fill = event.payload
        symbol = fill.get('symbol')
        quantity = fill.get('quantity', 0)
        price = fill.get('price', 0)
        direction = fill.get('direction', 'long')
        trade_id = event.correlation_id or fill.get('order_id')
        
        # Update or create position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'realized_pnl': 0,
                'entry_time': None,
                'entry_fills': []
            }
        
        position = self.positions[symbol]
        
        # Check if this is a closing trade
        is_closing = (
            (direction == 'long' and position['quantity'] < 0) or
            (direction == 'short' and position['quantity'] > 0) or
            (direction == 'sell' and position['quantity'] > 0) or
            (direction == 'cover' and position['quantity'] < 0)
        )
        
        if is_closing and position['quantity'] != 0:
            # Calculate P&L for the trade
            entry_price = position['avg_price']
            exit_price = price
            trade_quantity = min(abs(quantity), abs(position['quantity']))
            trade_direction = 'long' if position['quantity'] > 0 else 'short'
            
            # Update metrics
            self.metrics.update_from_trade(
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=trade_quantity,
                direction=trade_direction
            )
            
            # Store completed trade if enabled
            if self.store_trades:
                trade_record = {
                    'symbol': symbol,
                    'direction': trade_direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': trade_quantity,
                    'entry_time': position.get('entry_time'),
                    'exit_time': fill.get('timestamp'),
                    'pnl': (exit_price - entry_price) * trade_quantity 
                           if trade_direction == 'long' 
                           else (entry_price - exit_price) * trade_quantity
                }
                self.completed_trades.append(trade_record)
            
            # Clean up trade events if using trade_complete retention
            if self.retention_policy == 'trade_complete':
                # Remove events for completed portion
                self._cleanup_completed_trade_events(symbol, trade_id)
        
        # Update position
        if direction in ('long', 'buy'):
            # Opening or adding to long
            if position['quantity'] >= 0:
                # Adding to position
                total_value = position['quantity'] * position['avg_price'] + quantity * price
                position['quantity'] += quantity
                position['avg_price'] = total_value / position['quantity'] if position['quantity'] > 0 else 0
                if position['entry_time'] is None:
                    position['entry_time'] = fill.get('timestamp')
            else:
                # Closing short
                position['quantity'] += quantity
        else:  # short or sell
            # Opening or adding to short
            if position['quantity'] <= 0:
                # Adding to short position
                total_value = abs(position['quantity']) * position['avg_price'] + quantity * price
                position['quantity'] -= quantity
                position['avg_price'] = total_value / abs(position['quantity']) if position['quantity'] != 0 else 0
                if position['entry_time'] is None:
                    position['entry_time'] = fill.get('timestamp')
            else:
                # Closing long
                position['quantity'] -= quantity
        
        # Clear position if flat
        if abs(position['quantity']) < 1e-6:
            position['quantity'] = 0
            position['avg_price'] = 0
            position['entry_time'] = None
    
    def _process_portfolio_update(self, event: Event) -> None:
        """Process portfolio value update."""
        portfolio_value = event.payload.get('portfolio_value')
        timestamp = event.payload.get('timestamp') or event.timestamp
        
        if portfolio_value is not None:
            # Update metrics
            self.metrics.update_portfolio_value(portfolio_value, timestamp)
            
            # Handle equity curve snapshots
            if self.store_equity_curve:
                self.update_counter += 1
                if self.update_counter >= self.last_snapshot + self.snapshot_interval:
                    self.equity_curve.append({
                        'timestamp': timestamp.isoformat() if timestamp else None,
                        'value': portfolio_value,
                        'drawdown': self.metrics.current_drawdown,
                        'total_return': self.metrics.total_return
                    })
                    self.last_snapshot = self.update_counter
    
    def _process_position_update(self, event: Event) -> None:
        """Process position update event."""
        # Could be used for more detailed position tracking
        pass
    
    def _cleanup_completed_trade_events(self, symbol: str, trade_id: Optional[str]) -> None:
        """Remove events for completed trades to free memory.
        
        IMPORTANT: Only removes events for the specific trade_id (correlation_id).
        Each trade's events are tracked by correlation_id and only removed when
        THAT specific trade closes, not just any close.
        """
        if trade_id and trade_id in self.active_trades:
            del self.active_trades[trade_id]
        
        # In minimal retention mode, be more aggressive
        if self.retention_policy == 'minimal':
            # Only keep events for positions with open trades
            symbols_with_positions = {s for s, p in self.positions.items() 
                                    if p['quantity'] != 0}
            
            # Clear events for closed positions
            self.active_trades = {
                oid: events for oid, events in self.active_trades.items()
                if any(e.payload.get('symbol') in symbols_with_positions 
                      for e in events)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.get_metrics()
        
        # Add event tracking info
        metrics['events_retained'] = len(self.events) + sum(len(e) for e in self.active_trades.values())
        metrics['active_trades'] = len(self.active_trades)
        metrics['open_positions'] = sum(1 for p in self.positions.values() if p['quantity'] != 0)
        
        # Calculate objective function value
        metrics['objective_value'] = self._calculate_objective(metrics)
        metrics['objective_function'] = self.objective_config['name']
        
        # Add any custom metrics
        for custom_metric in self.custom_metrics:
            if callable(custom_metric):
                name = getattr(custom_metric, '__name__', 'custom')
                metrics[f'custom_{name}'] = custom_metric(metrics)
        
        return metrics
    
    def _calculate_objective(self, metrics: Dict[str, Any]) -> float:
        """Calculate objective function value for optimization."""
        func_name = self.objective_config['name']
        params = self.objective_config.get('params', {})
        
        # Built-in objective functions
        if func_name == 'sharpe_ratio':
            return metrics.get('sharpe_ratio', 0.0)
        elif func_name == 'total_return':
            return metrics.get('total_return', 0.0)
        elif func_name == 'profit_factor':
            return metrics.get('profit_factor', 0.0)
        elif func_name == 'win_rate':
            return metrics.get('win_rate', 0.0)
        elif func_name == 'calmar_ratio':
            # Return / MaxDrawdown (but we don't have drawdown in minimal mode)
            if metrics.get('max_drawdown', 0) > 0:
                return metrics.get('total_return', 0) / metrics.get('max_drawdown', 0)
            return 0.0
        elif func_name == 'sortino_ratio':
            # Would need downside deviation tracking
            return metrics.get('sharpe_ratio', 0.0) * params.get('sortino_multiplier', 1.2)
        elif func_name == 'weighted_combination':
            # Weighted combination of multiple metrics
            weights = params.get('weights', {
                'sharpe_ratio': 0.4,
                'win_rate': 0.3,
                'profit_factor': 0.3
            })
            value = 0.0
            for metric, weight in weights.items():
                value += metrics.get(metric, 0.0) * weight
            return value
        else:
            # Unknown objective, default to Sharpe
            logger.warning(f"Unknown objective function: {func_name}, using Sharpe ratio")
            return metrics.get('sharpe_ratio', 0.0)
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete results including metrics, trades, and equity curve."""
        results = {
            'metrics': self.get_metrics(),
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': self.metrics.current_value,
                'total_return': self.metrics.total_return,
                'sharpe_ratio': self.metrics.get_metrics()['sharpe_ratio'],
                'max_drawdown': self.metrics.max_drawdown,
                'total_trades': self.metrics.n_trades,
                'win_rate': self.metrics.get_metrics()['win_rate']
            }
        }
        
        # Add trades if stored
        if self.store_trades and self.completed_trades:
            results['trades'] = self.completed_trades
        
        # Add equity curve if stored
        if self.store_equity_curve and self.equity_curve:
            results['equity_curve'] = self.equity_curve
        
        return results
    
    def clear(self) -> None:
        """Clear all retained events and reset metrics."""
        self.events.clear()
        self.active_trades.clear()
        self.completed_trades.clear()
        self.positions.clear()
        self.equity_curve.clear()
        
        # Reset metrics
        self.metrics = StreamingMetrics(
            initial_capital=self.initial_capital,
            current_value=self.initial_capital
        )