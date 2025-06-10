"""
Metrics observer for event-based performance tracking.

Implements memory-efficient metrics collection using trade-complete retention policy.
Only holds events for open trades, prunes when trades close.
"""

from typing import Dict, Any, Optional, List, Deque, Set
from collections import deque, defaultdict
from datetime import datetime
import logging

from ..protocols import EventObserverProtocol, MetricsCalculatorProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


class BasicMetricsCalculator:
    """
    Basic implementation of MetricsCalculatorProtocol.
    
    Calculates essential trading metrics: returns, win rate, P&L.
    """
    
    def __init__(self, initial_capital: float = 100000.0, annualization_factor: float = 252.0):
        """
        Initialize calculator.
        
        Args:
            initial_capital: Starting capital
            annualization_factor: Factor for annualizing returns (252 for daily)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.annualization_factor = annualization_factor
        
        # Trade tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Portfolio tracking
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        
        # Return tracking
        self.returns: List[float] = []
        
    def update_from_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str
    ) -> None:
        """Update metrics from a completed trade."""
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity
        
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
        
        # Update capital
        self.current_capital += pnl
        
        # Track return
        if self.current_capital > 0:
            ret = pnl / (self.current_capital - pnl)
            self.returns.append(ret)
    
    def update_portfolio_value(
        self,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update metrics with new portfolio value."""
        self.current_capital = value
        
        # Update peak and drawdown
        if value > self.peak_value:
            self.peak_value = value
        else:
            drawdown = (self.peak_value - value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf'),
            'max_drawdown': self.max_drawdown,
            'current_capital': self.current_capital
        }
        
        # Calculate Sharpe ratio if we have returns
        if len(self.returns) > 1:
            import math
            mean_return = sum(self.returns) / len(self.returns)
            variance = sum((r - mean_return) ** 2 for r in self.returns) / (len(self.returns) - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0
            sharpe = (mean_return / std_dev * math.sqrt(self.annualization_factor)) if std_dev > 0 else 0.0
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0.0
            
        return metrics


class MetricsObserver:
    """
    Event observer that calculates metrics with memory-efficient retention.
    
    Implements trade-complete retention policy:
    - Only keeps events for open trades
    - Prunes events when trades close
    - Calculates metrics incrementally
    """
    
    def __init__(
        self,
        calculator: MetricsCalculatorProtocol,
        retention_policy: str = 'trade_complete',
        max_events: int = 1000
    ):
        """
        Initialize metrics observer.
        
        Args:
            calculator: Metrics calculator instance
            retention_policy: How to manage event memory ('trade_complete', 'all', 'none')
            max_events: Maximum events to retain per trade
        """
        self.calculator = calculator
        self.retention_policy = retention_policy
        self.max_events = max_events
        
        # Trade tracking for retention policy
        self.active_trades: Dict[str, List[Event]] = {}  # correlation_id -> events
        self.completed_trades: List[Dict[str, Any]] = []
        
        # Event statistics
        self.events_observed = 0
        self.events_pruned = 0
        
        # Position tracking for trade matching
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        
        logger.info(f"MetricsObserver initialized with retention_policy='{retention_policy}'")
    
    def on_publish(self, event: Event) -> None:
        """Called when event is published."""
        self.events_observed += 1
        
        # Process based on event type
        if event.event_type == EventType.POSITION_OPEN:
            self._handle_position_open(event)
        elif event.event_type == EventType.FILL:
            self._handle_fill(event)
        elif event.event_type == EventType.POSITION_CLOSE:
            self._handle_position_close(event)
        elif event.event_type == EventType.PORTFOLIO_UPDATE:
            self._handle_portfolio_update(event)
    
    def on_delivered(self, event: Event, handler: Any) -> None:
        """Called when event is delivered to handler."""
        pass  # Not needed for metrics
    
    def on_error(self, event: Event, handler: Any, error: Exception) -> None:
        """Called when handler raises error."""
        logger.warning(f"Handler error for event {event.event_type}: {error}")
    
    def _handle_position_open(self, event: Event) -> None:
        """Handle position open event."""
        correlation_id = event.correlation_id
        if not correlation_id:
            return
            
        # Start tracking this trade
        if self.retention_policy == 'trade_complete':
            if correlation_id not in self.active_trades:
                self.active_trades[correlation_id] = []
            self.active_trades[correlation_id].append(event)
            
            # Enforce max events per trade
            if len(self.active_trades[correlation_id]) > self.max_events:
                self.events_pruned += 1
                self.active_trades[correlation_id].pop(0)
        
        # Update position tracking
        symbol = event.payload.get('symbol')
        if symbol:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'entry_events': []
                }
            self.positions[symbol]['entry_events'].append(event)
    
    def _handle_fill(self, event: Event) -> None:
        """Handle fill event."""
        correlation_id = event.correlation_id
        
        # Track with active trade if using retention policy
        if correlation_id and self.retention_policy == 'trade_complete':
            if correlation_id not in self.active_trades:
                self.active_trades[correlation_id] = []
            self.active_trades[correlation_id].append(event)
            
            # Enforce max events
            if len(self.active_trades[correlation_id]) > self.max_events:
                self.events_pruned += 1
                self.active_trades[correlation_id].pop(0)
    
    def _handle_position_close(self, event: Event) -> None:
        """Handle position close event."""
        correlation_id = event.correlation_id
        
        # Extract trade info
        payload = event.payload
        symbol = payload.get('symbol')
        quantity = payload.get('quantity', 0)
        price = payload.get('price', 0)
        pnl = payload.get('pnl', 0)
        
        # Update metrics if we have position info
        if symbol and symbol in self.positions:
            position = self.positions[symbol]
            if position.get('avg_price', 0) > 0:
                # Determine trade direction from position
                direction = 'long' if position['quantity'] > 0 else 'short'
                
                # Update calculator
                self.calculator.update_from_trade(
                    entry_price=position['avg_price'],
                    exit_price=price,
                    quantity=abs(quantity),
                    direction=direction
                )
            
            # Clear position if flat
            position['quantity'] -= quantity
            if abs(position['quantity']) < 1e-6:
                del self.positions[symbol]
        
        # Prune events for completed trade
        if correlation_id and self.retention_policy == 'trade_complete':
            if correlation_id in self.active_trades:
                # Store completed trade info if needed
                trade_events = self.active_trades[correlation_id]
                self.completed_trades.append({
                    'correlation_id': correlation_id,
                    'symbol': symbol,
                    'pnl': pnl,
                    'event_count': len(trade_events),
                    'close_time': event.timestamp
                })
                
                # Prune events
                self.events_pruned += len(trade_events)
                del self.active_trades[correlation_id]
                
                logger.debug(f"Pruned {len(trade_events)} events for completed trade {correlation_id}")
    
    def _handle_portfolio_update(self, event: Event) -> None:
        """Handle portfolio update event."""
        value = event.payload.get('portfolio_value')
        timestamp = event.payload.get('timestamp', event.timestamp)
        
        if value is not None:
            self.calculator.update_portfolio_value(value, timestamp)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics including observer statistics."""
        metrics = self.calculator.get_metrics()
        
        # Add observer statistics
        metrics['observer_stats'] = {
            'events_observed': self.events_observed,
            'events_pruned': self.events_pruned,
            'active_trades': len(self.active_trades),
            'retention_policy': self.retention_policy,
            'memory_efficiency': self.events_pruned / self.events_observed if self.events_observed > 0 else 0.0
        }
        
        # Wrap calculator metrics in 'metrics' key
        return {
            'metrics': metrics,
            'observer_stats': metrics['observer_stats']
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete results including trades and metrics."""
        return {
            'metrics': self.calculator.get_metrics(),
            'observer_stats': {
                'events_observed': self.events_observed,
                'events_pruned': self.events_pruned,
                'active_trades': len(self.active_trades),
                'completed_trades': len(self.completed_trades)
            },
            'trades': self.completed_trades[:100]  # Limit to recent 100 trades
        }