# MetricsEventTracer Refactoring to Observer Pattern

# ============================================
# ANALYSIS OF CURRENT IMPLEMENTATION
# ============================================
"""
Current MetricsEventTracer Strengths:
1. Memory Efficient - Smart retention policies that prune completed trades
2. Purpose-Built - Specifically designed for portfolio metrics calculation
3. Trade Tracking - Correlates open/close events effectively

Issue to Address:
Your MetricsEventTracer is doing two things:
1. Observing events (following the EventObserver pattern)
2. Calculating metrics (business logic)

This refactoring separates these concerns following Protocol + Composition.
"""

# ============================================
# File: src/core/events/protocols.py
# ============================================
"""Add metrics calculator protocol to existing protocols."""

from typing import Protocol, Dict, Any, Optional
from datetime import datetime

class MetricsCalculatorProtocol(Protocol):
    """Protocol for metrics calculation."""
    
    def update_from_trade(
        self, 
        entry_price: float, 
        exit_price: float, 
        quantity: float, 
        direction: str
    ) -> None:
        """Update metrics from a completed trade."""
        ...
    
    def update_portfolio_value(
        self, 
        value: float, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update metrics with new portfolio value."""
        ...
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        ...


# ============================================
# File: src/core/events/observers/metrics.py
# ============================================
"""Metrics observer implementation using Protocol + Composition."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..protocols import EventObserverProtocol, MetricsCalculatorProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class MetricsObserver(EventObserverProtocol):
    """
    Pure observer that delegates to metrics calculator.
    
    This follows Protocol + Composition approach perfectly!
    Observes events and delegates calculation to a separate calculator.
    """
    
    # Composed calculator - not inherited!
    calculator: MetricsCalculatorProtocol
    
    # Configuration
    retention_policy: str = "trade_complete"
    max_events: int = 1000
    
    # Temporary storage for correlation
    active_trades: Dict[str, List[Event]] = field(default_factory=dict)
    _event_count: int = 0
    _pruned_count: int = 0
    
    def on_publish(self, event: Event) -> None:
        """Observe published events."""
        self._event_count += 1
        
        # Store events for active trades based on retention policy
        if self.retention_policy == "trade_complete":
            self._store_trade_event(event)
        elif self.retention_policy == "sliding_window":
            # Different retention strategy
            pass
        
        # Process specific events
        if event.event_type == EventType.FILL.value:
            self._process_fill(event)
        elif event.event_type == EventType.PORTFOLIO_UPDATE.value:
            self._process_portfolio_update(event)
        elif event.event_type == EventType.POSITION_CLOSE.value:
            self._process_position_close(event)
        elif event.event_type == EventType.POSITION_OPEN.value:
            self._process_position_open(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Not needed for metrics observation."""
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Log errors but don't affect metrics."""
        logger.warning(f"Error processing {event.event_type}: {error}")
    
    def _store_trade_event(self, event: Event) -> None:
        """Store event if it's part of an active trade."""
        trade_id = event.correlation_id
        
        if not trade_id:
            return
        
        # Events we care about for trade tracking
        relevant_types = [
            EventType.POSITION_OPEN.value,
            EventType.ORDER_REQUEST.value,
            EventType.ORDER.value,
            EventType.FILL.value,
            EventType.POSITION_CLOSE.value
        ]
        
        if event.event_type in relevant_types:
            if trade_id not in self.active_trades:
                self.active_trades[trade_id] = []
            self.active_trades[trade_id].append(event)
    
    def _process_position_open(self, event: Event) -> None:
        """Track new position."""
        # Ensure we're tracking this trade
        trade_id = event.correlation_id
        if trade_id and trade_id not in self.active_trades:
            self.active_trades[trade_id] = [event]
    
    def _process_fill(self, event: Event) -> None:
        """Process fill event."""
        # For now, just ensure it's tracked
        # Actual P&L calculation happens on position close
        pass
    
    def _process_portfolio_update(self, event: Event) -> None:
        """Process portfolio value update."""
        portfolio_value = event.payload.get('portfolio_value')
        timestamp = event.payload.get('timestamp') or event.timestamp
        
        if portfolio_value is not None:
            self.calculator.update_portfolio_value(portfolio_value, timestamp)
    
    def _process_position_close(self, event: Event) -> None:
        """Process position close and update metrics."""
        trade_id = event.correlation_id
        
        if trade_id in self.active_trades:
            # Find the opening event
            events = self.active_trades[trade_id]
            open_event = next(
                (e for e in events if e.event_type == EventType.POSITION_OPEN.value), 
                None
            )
            
            if open_event:
                # Extract data from events
                entry_price = open_event.payload.get('price', 0)
                exit_price = event.payload.get('price', 0)
                quantity = event.payload.get('quantity', 0)
                direction = open_event.payload.get('direction', 'long')
                
                # Delegate calculation to calculator
                self.calculator.update_from_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    direction=direction
                )
            
            # Apply retention policy
            if self.retention_policy == "trade_complete":
                # Remove all events for this completed trade
                pruned = len(self.active_trades[trade_id])
                del self.active_trades[trade_id]
                self._pruned_count += pruned
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from calculator plus observer stats."""
        return {
            'metrics': self.calculator.get_metrics(),
            'observer_stats': {
                'events_observed': self._event_count,
                'events_pruned': self._pruned_count,
                'active_trades': len(self.active_trades),
                'retention_policy': self.retention_policy
            }
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete results (alias for container compatibility)."""
        return self.get_metrics()


# ============================================
# Specialized Observers for Different Metrics
# ============================================

@dataclass
class SharpeRatioObserver(EventObserverProtocol):
    """Observer focused only on Sharpe ratio calculation."""
    
    calculator: 'SharpeRatioCalculator'
    
    def on_publish(self, event: Event) -> None:
        """Only process portfolio updates for returns."""
        if event.event_type == EventType.PORTFOLIO_UPDATE.value:
            value = event.payload.get('portfolio_value')
            if value:
                self.calculator.add_portfolio_value(value)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


@dataclass
class DrawdownObserver(EventObserverProtocol):
    """Observer focused on drawdown tracking."""
    
    calculator: 'DrawdownCalculator'
    
    def on_publish(self, event: Event) -> None:
        """Track portfolio values for drawdown."""
        if event.event_type == EventType.PORTFOLIO_UPDATE.value:
            value = event.payload.get('portfolio_value')
            if value:
                self.calculator.update_value(value)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


# ============================================
# Benefits of This Approach
# ============================================
"""
1. **Separation of Concerns**:
   - MetricsObserver handles event observation and retention
   - StreamingMetrics (your existing class) handles calculations
   - Clean, single-responsibility components

2. **Reusable**:
   ```python
   # Can use same observer with different calculators
   sharpe_calculator = SharpeRatioCalculator()
   sharpe_observer = MetricsObserver(calculator=sharpe_calculator)
   
   drawdown_calculator = DrawdownCalculator()
   drawdown_observer = MetricsObserver(calculator=drawdown_calculator)
   ```

3. **Testable**:
   ```python
   # Easy to test with mock calculator
   mock_calc = MockMetricsCalculator()
   observer = MetricsObserver(calculator=mock_calc)
   observer.on_publish(test_event)
   assert mock_calc.update_called
   ```

4. **Composable**:
   ```python
   # Can attach multiple observers for different metrics
   event_bus.attach_observer(sharpe_observer)
   event_bus.attach_observer(drawdown_observer)
   event_bus.attach_observer(win_rate_observer)
   ```

5. **Protocol + Composition**:
   - No inheritance needed
   - Any object implementing MetricsCalculatorProtocol works
   - Observers are composed with calculators, not inherited
"""

# ============================================
# Integration Example
# ============================================
"""
How to integrate this with your existing system:
"""

# Your existing StreamingMetrics class already implements the protocol!
from ..containers.metrics import StreamingMetrics

class PortfolioContainer:
    def _setup_event_tracing_metrics(self):
        """Setup metrics using observer pattern."""
        
        # Create calculator (your existing StreamingMetrics)
        calculator = StreamingMetrics(
            initial_capital=self.config.config.get('initial_capital', 100000.0),
            annualization_factor=252.0
        )
        
        # Create observer with the calculator
        metrics_observer = MetricsObserver(
            calculator=calculator,
            retention_policy=self.config.config.get('retention_policy', 'trade_complete'),
            max_events=self.config.config.get('max_events', 1000)
        )
        
        # Attach observer to event bus
        self.event_bus.attach_observer(metrics_observer)
        
        # Store reference for compatibility
        self.streaming_metrics = metrics_observer
        
        logger.info(f"Metrics observer attached to {self.container_id}")
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from observer."""
        if self.streaming_metrics:
            return self.streaming_metrics.get_metrics()
        return None


# ============================================
# Clean Observer-Based Metrics Design (Best Practice)
# ============================================
"""
A clean metrics observer implementation following best practices.
This design is superior to legacy approaches because it:
- Follows single responsibility principle
- Uses immutable state where possible
- Provides clear separation of concerns
- Is highly composable and testable
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging
import numpy as np

from ..protocols import EventObserverProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """
    Immutable trade metrics using functional approach.
    
    Instead of mutating state, we return new instances.
    """
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    def with_trade(self, pnl: float) -> 'TradeMetrics':
        """Return new metrics with trade added."""
        return TradeMetrics(
            total_trades=self.total_trades + 1,
            winning_trades=self.winning_trades + (1 if pnl > 0 else 0),
            total_pnl=self.total_pnl + pnl,
            gross_profit=self.gross_profit + max(pnl, 0),
            gross_loss=self.gross_loss + min(pnl, 0)
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with calculated fields."""
        total = max(self.total_trades, 1)  # Avoid division by zero
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': self.winning_trades / total,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': abs(self.gross_loss),
            'profit_factor': abs(self.gross_profit / self.gross_loss) if self.gross_loss < 0 else float('inf'),
            'average_win': self.gross_profit / max(self.winning_trades, 1),
            'average_loss': abs(self.gross_loss) / max(self.total_trades - self.winning_trades, 1)
        }


@dataclass
class PositionTracker:
    """
    Tracks open positions for metrics calculation.
    
    Separate class for position management following SRP.
    """
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def open_position(self, correlation_id: str, entry_price: float, 
                     quantity: int, symbol: str, timestamp: datetime, 
                     direction: str = 'long') -> None:
        """Track new position."""
        self.positions[correlation_id] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'symbol': symbol,
            'entry_time': timestamp,
            'direction': direction
        }
    
    def close_position(self, correlation_id: str, exit_price: float, 
                      timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Close position and return trade data."""
        if correlation_id not in self.positions:
            logger.warning(f"Attempting to close unknown position: {correlation_id}")
            return None
            
        position = self.positions.pop(correlation_id)
        
        # Calculate PnL based on direction
        quantity = position['quantity']
        entry = position['entry_price']
        
        if position['direction'] == 'long':
            pnl = (exit_price - entry) * quantity
        else:  # short
            pnl = (entry - exit_price) * quantity
        
        return {
            'pnl': pnl,
            'pnl_pct': pnl / (entry * quantity) if entry > 0 else 0,
            'entry_price': entry,
            'exit_price': exit_price,
            'quantity': quantity,
            'symbol': position['symbol'],
            'direction': position['direction'],
            'duration_seconds': (timestamp - position['entry_time']).total_seconds()
        }
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current open positions."""
        return self.positions.copy()


@dataclass
class EquityCurveTracker:
    """
    Tracks portfolio value over time with configurable resolution.
    
    Separate class for equity curve management.
    """
    max_points: int = 1000
    track_returns: bool = True
    
    _values: List[float] = field(default_factory=list)
    _timestamps: List[datetime] = field(default_factory=list)
    _returns: List[float] = field(default_factory=list)
    
    def update(self, value: float, timestamp: datetime) -> None:
        """Update equity curve with new value."""
        # Calculate return if we have previous value
        if self._values and self.track_returns:
            prev_value = self._values[-1]
            if prev_value > 0:
                ret = (value - prev_value) / prev_value
                self._returns.append(ret)
        
        # Add new value
        self._values.append(value)
        self._timestamps.append(timestamp)
        
        # Prune old points
        if len(self._values) > self.max_points:
            self._values.pop(0)
            self._timestamps.pop(0)
            if self._returns:
                self._returns.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate equity curve statistics."""
        if len(self._values) < 2:
            return {
                'current_value': self._values[-1] if self._values else 0,
                'peak_value': self._values[-1] if self._values else 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'volatility': 0
            }
        
        values = np.array(self._values)
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        
        stats = {
            'current_value': values[-1],
            'peak_value': peak[-1],
            'max_drawdown': np.max(drawdowns),
            'current_drawdown': drawdowns[-1]
        }
        
        # Add return statistics if available
        if self._returns and len(self._returns) >= 20:
            returns = np.array(self._returns)
            stats['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            stats['volatility'] = returns.std() * np.sqrt(252)
            stats['avg_return'] = returns.mean()
        else:
            stats['sharpe_ratio'] = 0
            stats['volatility'] = 0
            stats['avg_return'] = 0
        
        return stats


@dataclass
class MetricsObserver(EventObserverProtocol):
    """
    Clean metrics observer following best practices.
    
    Key design principles:
    - Single responsibility: Only observes and calculates metrics
    - Immutable state where possible (TradeMetrics)
    - Composition over inheritance (uses PositionTracker, EquityCurveTracker)
    - Memory efficient with automatic pruning
    - Protocol compliant for easy integration
    """
    
    # Configuration
    track_equity_curve: bool = True
    track_trade_history: bool = False
    max_equity_points: int = 1000
    initial_capital: float = 100000.0
    
    # Composed components
    _metrics: TradeMetrics = field(default_factory=TradeMetrics)
    _positions: PositionTracker = field(default_factory=PositionTracker)
    _equity_tracker: EquityCurveTracker = field(init=False)
    
    # Optional trade history
    _trade_history: List[Dict[str, Any]] = field(default_factory=list)
    _max_trade_history: int = 1000
    
    def __post_init__(self):
        """Initialize equity tracker with config."""
        self._equity_tracker = EquityCurveTracker(
            max_points=self.max_equity_points,
            track_returns=True
        )
        # Initialize with starting capital
        self._equity_tracker.update(self.initial_capital, datetime.now())
    
    def on_publish(self, event: Event) -> None:
        """Process events for metrics calculation."""
        # Pattern match on event types
        if event.event_type == EventType.POSITION_OPEN:
            self._handle_position_open(event)
            
        elif event.event_type == EventType.POSITION_CLOSE:
            self._handle_position_close(event)
            
        elif event.event_type == EventType.PORTFOLIO_UPDATE:
            self._handle_portfolio_update(event)
            
        elif event.event_type == EventType.FILL:
            # Alternative: Handle fills if positions aren't explicitly opened/closed
            self._handle_fill(event)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Not needed for metrics observation."""
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Log errors but don't affect metrics."""
        logger.warning(f"Error in handler {handler.__name__}: {error}")
    
    def _handle_position_open(self, event: Event) -> None:
        """Track new position opening."""
        payload = event.payload
        
        self._positions.open_position(
            correlation_id=event.correlation_id or str(uuid.uuid4()),
            entry_price=payload.get('price', 0),
            quantity=payload.get('quantity', 0),
            symbol=payload.get('symbol', ''),
            timestamp=event.timestamp,
            direction=payload.get('direction', 'long')
        )
        
        logger.debug(f"Position opened: {payload.get('symbol')} @ {payload.get('price')}")
    
    def _handle_position_close(self, event: Event) -> None:
        """Handle position closing and update metrics."""
        payload = event.payload
        
        # Close position and get trade data
        trade_data = self._positions.close_position(
            correlation_id=event.correlation_id,
            exit_price=payload.get('price', 0),
            timestamp=event.timestamp
        )
        
        if trade_data:
            # Update immutable metrics
            self._metrics = self._metrics.with_trade(trade_data['pnl'])
            
            # Store trade history if enabled
            if self.track_trade_history:
                self._trade_history.append(trade_data)
                if len(self._trade_history) > self._max_trade_history:
                    self._trade_history.pop(0)
            
            logger.debug(
                f"Trade closed: {trade_data['symbol']} "
                f"PnL=${trade_data['pnl']:.2f} ({trade_data['pnl_pct']:.2%})"
            )
    
    def _handle_portfolio_update(self, event: Event) -> None:
        """Update portfolio value and equity curve."""
        payload = event.payload
        value = payload.get('portfolio_value', payload.get('total_value'))
        
        if value and self.track_equity_curve:
            self._equity_tracker.update(value, event.timestamp)
    
    def _handle_fill(self, event: Event) -> None:
        """
        Handle fill events for systems that don't use explicit position events.
        
        This is an alternative approach where we infer positions from fills.
        """
        # Implementation depends on your system's fill event structure
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics snapshot."""
        # Get base trade metrics
        metrics = self._metrics.to_dict()
        
        # Add equity curve statistics
        if self.track_equity_curve:
            metrics.update(self._equity_tracker.get_statistics())
        
        # Add position information
        metrics['open_positions'] = len(self._positions.positions)
        metrics['open_position_details'] = self._positions.get_open_positions()
        
        # Add trade history if tracked
        if self.track_trade_history:
            metrics['recent_trades'] = self._trade_history[-10:]  # Last 10 trades
            metrics['trade_history_size'] = len(self._trade_history)
        
        return metrics
    
    def get_equity_curve(self) -> List[float]:
        """Get equity curve values."""
        return self._equity_tracker._values.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history if tracked."""
        return self._trade_history.copy()


# ============================================
# Specialized Observers for Specific Metrics
# ============================================
"""
Following the single responsibility principle, you can create
specialized observers for specific metrics that can be composed
together.
"""

@dataclass
class SharpeRatioObserver(EventObserverProtocol):
    """
    Observer focused solely on Sharpe ratio calculation.
    
    Minimal memory footprint - only tracks returns.
    """
    
    window: int = 252
    risk_free_rate: float = 0.0
    
    _returns: List[float] = field(default_factory=list)
    _last_value: Optional[float] = None
    
    def on_publish(self, event: Event) -> None:
        """Only process portfolio value updates."""
        if event.event_type == EventType.PORTFOLIO_UPDATE:
            value = event.payload.get('portfolio_value', event.payload.get('total_value'))
            
            if self._last_value and value:
                # Calculate return
                ret = (value - self._last_value) / self._last_value
                self._returns.append(ret)
                
                # Maintain window
                if len(self._returns) > self.window:
                    self._returns.pop(0)
            
            self._last_value = value
    
    def get_sharpe_ratio(self) -> float:
        """Calculate current Sharpe ratio."""
        if len(self._returns) < 20:  # Need minimum data
            return 0.0
        
        returns = np.array(self._returns)
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free
        
        if returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        return 0.0
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


@dataclass
class DrawdownObserver(EventObserverProtocol):
    """
    Observer focused on drawdown tracking.
    
    Tracks peak values and current drawdowns.
    """
    
    _peak_value: float = 0.0
    _current_value: float = 0.0
    _max_drawdown: float = 0.0
    _drawdown_start: Optional[datetime] = None
    _current_drawdown_start: Optional[datetime] = None
    
    def on_publish(self, event: Event) -> None:
        """Track portfolio values for drawdown calculation."""
        if event.event_type == EventType.PORTFOLIO_UPDATE:
            value = event.payload.get('portfolio_value', event.payload.get('total_value'))
            
            if value:
                self._current_value = value
                
                # Update peak
                if value > self._peak_value:
                    self._peak_value = value
                    self._current_drawdown_start = None
                else:
                    # We're in a drawdown
                    if self._current_drawdown_start is None:
                        self._current_drawdown_start = event.timestamp
                    
                    current_dd = (self._peak_value - value) / self._peak_value
                    if current_dd > self._max_drawdown:
                        self._max_drawdown = current_dd
                        self._drawdown_start = self._current_drawdown_start
    
    def get_drawdown_metrics(self) -> Dict[str, Any]:
        """Get current drawdown metrics."""
        current_dd = 0.0
        if self._peak_value > 0:
            current_dd = (self._peak_value - self._current_value) / self._peak_value
        
        return {
            'max_drawdown': self._max_drawdown,
            'current_drawdown': current_dd,
            'peak_value': self._peak_value,
            'current_value': self._current_value,
            'in_drawdown': current_dd > 0,
            'drawdown_start': self._drawdown_start,
            'current_drawdown_start': self._current_drawdown_start
        }
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        pass


# ============================================
# Composing Multiple Observers
# ============================================
"""
The beauty of this approach is that you can compose multiple
specialized observers together for comprehensive metrics.
"""

class PortfolioContainer:
    """Example of using multiple observers together."""
    
    def __init__(self, config):
        self.container_id = config['id']
        self.event_bus = EventBus(self.container_id)
        
        # Core metrics observer
        self.metrics_observer = MetricsObserver(
            track_equity_curve=config.get('track_equity', True),
            track_trade_history=config.get('track_trades', False),
            initial_capital=config.get('initial_capital', 100000.0)
        )
        
        # Specialized observers
        self.sharpe_observer = SharpeRatioObserver(
            window=config.get('sharpe_window', 252)
        )
        
        self.drawdown_observer = DrawdownObserver()
        
        # Attach all observers
        self.event_bus.attach_observer(self.metrics_observer)
        self.event_bus.attach_observer(self.sharpe_observer)
        self.event_bus.attach_observer(self.drawdown_observer)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from all observers."""
        metrics = self.metrics_observer.get_metrics()
        metrics['sharpe_ratio'] = self.sharpe_observer.get_sharpe_ratio()
        metrics.update(self.drawdown_observer.get_drawdown_metrics())
        return metrics


# ============================================
# Benefits of Clean Observer Design
# ============================================
"""
1. **Single Responsibility**: Each observer has one clear purpose
2. **Composable**: Mix and match observers as needed
3. **Testable**: Each observer can be tested in isolation
4. **Memory Efficient**: Only track what you need
5. **Immutable State**: Functional approach where appropriate
6. **Protocol Compliant**: Works with any EventObserverProtocol system

This design is superior to monolithic approaches because:
- No god objects doing everything
- Clear separation of concerns
- Easy to add new metrics without touching existing code
- Can run different observers in different environments
- Observers are lightweight and focused
"""


# ============================================
# Memory Consumption Observer
# ============================================
"""
A genuinely useful observer that tracks memory consumption by monitoring
event flow and container internals. This enables granular diagnostics
on system performance.
"""

import gc
import sys
import weakref
import tracemalloc
from collections import deque
import random

@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot."""
    timestamp: datetime
    container_id: str
    event_count: int
    memory_bytes: int
    tracer_events: int
    open_positions: int
    gc_stats: Optional[Dict[str, Any]] = None
    top_allocations: Optional[List[Tuple[str, int]]] = None


@dataclass
class ContainerMemoryObserver(EventObserverProtocol):
    """
    Memory observer that tracks container memory consumption.
    
    Key insights:
    1. Events carry references to the objects creating them
    2. Different event types have different memory implications
    3. Can access container internals through weak references
    """
    
    # Configuration
    sampling_rate: float = 0.01  # Sample 1% of events
    deep_inspection_interval: int = 1000
    track_allocations: bool = False
    emit_warnings: bool = True
    memory_threshold_mb: float = 100.0
    
    # Internal state
    container_refs: Dict[str, weakref.ref] = field(default_factory=dict)
    memory_snapshots: deque = field(default_factory=lambda: deque(maxlen=1000))
    event_counts: Dict[str, int] = field(default_factory=dict)
    memory_by_type: Dict[str, int] = field(default_factory=dict)
    _last_warning_time: Optional[datetime] = None
    
    def attach_to_container(self, container) -> None:
        """
        Called when attached to a container's event bus.
        
        This is the key - we get a reference to the container itself!
        """
        self.container_refs[container.container_id] = weakref.ref(container)
        
        if self.track_allocations:
            tracemalloc.start()
    
    def on_publish(self, event: Event) -> None:
        """Monitor memory through event patterns."""
        container_id = event.container_id or "unknown"
        self.event_counts[container_id] = self.event_counts.get(container_id, 0) + 1
        
        # Estimate memory impact of this event type
        memory_impact = self._estimate_memory_impact(event)
        event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        self.memory_by_type[event_type] = self.memory_by_type.get(event_type, 0) + memory_impact
        
        # Periodic sampling
        if random.random() < self.sampling_rate:
            self._sample_memory(container_id)
        
        # Deep inspection at intervals
        if self.event_counts[container_id] % self.deep_inspection_interval == 0:
            self._deep_memory_inspection(container_id)
        
        # Check for memory anomalies
        if event.event_type == EventType.POSITION_CLOSE:
            # Memory should drop after position close
            self._check_memory_freed(container_id)
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Not needed for memory tracking."""
        pass
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Track errors that might indicate memory issues."""
        if "memory" in str(error).lower():
            self._emit_memory_warning(event.container_id, f"Memory-related error: {error}")
    
    def _estimate_memory_impact(self, event: Event) -> int:
        """
        Estimate memory impact of different event types.
        
        This helps identify which events are memory-heavy.
        """
        # Base size
        base_size = sys.getsizeof(event)
        payload_size = len(str(event.payload)) * 2 if event.payload else 0
        
        # Event-specific estimates
        if event.event_type == EventType.POSITION_OPEN:
            # Opening positions increases memory
            return base_size + payload_size + 1000
            
        elif event.event_type == EventType.POSITION_CLOSE:
            # Closing should free memory (negative impact)
            return -1000
            
        elif event.event_type == EventType.BAR:
            # Market data is usually transient
            return base_size + payload_size
            
        elif event.event_type == EventType.FEATURES:
            # Features can be memory-heavy
            return base_size + payload_size + 500
            
        return base_size + payload_size
    
    def _sample_memory(self, container_id: str) -> None:
        """Quick memory sample."""
        if container_id in self.container_refs:
            container = self.container_refs[container_id]()
            if container:
                # Quick memory check
                memory_bytes = self._get_container_memory_usage(container)
                
                # Check threshold
                if self.emit_warnings and memory_bytes > self.memory_threshold_mb * 1024 * 1024:
                    self._emit_memory_warning(container_id, f"Memory usage exceeds {self.memory_threshold_mb}MB")
    
    def _deep_memory_inspection(self, container_id: str) -> None:
        """Detailed memory inspection of container."""
        if container_id not in self.container_refs:
            return
            
        container = self.container_refs[container_id]()
        if not container:
            return
        
        # Collect detailed metrics
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            container_id=container_id,
            event_count=self.event_counts.get(container_id, 0),
            memory_bytes=self._get_container_memory_usage(container),
            tracer_events=self._get_tracer_event_count(container),
            open_positions=self._get_open_position_count(container)
        )
        
        # GC stats
        snapshot.gc_stats = {
            'collections': gc.get_count(),
            'collected': gc.collect(0),  # Collect generation 0
            'uncollectable': len(gc.garbage)
        }
        
        # Top memory allocations if tracking
        if self.track_allocations and tracemalloc.is_tracing():
            snapshot.top_allocations = self._get_top_allocations()
        
        self.memory_snapshots.append(snapshot)
        
        # Emit as event for downstream analysis
        self._emit_memory_snapshot_event(snapshot)
    
    def _get_container_memory_usage(self, container) -> int:
        """Estimate container's memory usage."""
        total = 0
        
        # Event bus and its contents
        if hasattr(container, 'event_bus'):
            total += sys.getsizeof(container.event_bus)
            
            # Tracer memory if enabled
            if hasattr(container.event_bus, '_tracer') and container.event_bus._tracer:
                tracer = container.event_bus._tracer
                if hasattr(tracer, 'events'):
                    total += sys.getsizeof(tracer.events)
        
        # Metrics memory
        if hasattr(container, 'metrics'):
            total += sys.getsizeof(container.metrics)
            if hasattr(container.metrics, 'metrics'):
                total += sys.getsizeof(container.metrics.metrics)
        
        # Open positions
        if hasattr(container, 'open_positions'):
            total += sys.getsizeof(container.open_positions)
            for pos in container.open_positions.values():
                total += sys.getsizeof(pos)
        
        return total
    
    def _get_tracer_event_count(self, container) -> int:
        """Get number of events in tracer."""
        if hasattr(container, 'event_bus') and hasattr(container.event_bus, '_tracer'):
            tracer = container.event_bus._tracer
            if tracer and hasattr(tracer, 'events'):
                return len(tracer.events)
        return 0
    
    def _get_open_position_count(self, container) -> int:
        """Get number of open positions."""
        if hasattr(container, 'open_positions'):
            return len(container.open_positions)
        return 0
    
    def _get_top_allocations(self) -> List[Tuple[str, int]]:
        """Get top memory allocations."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return [(str(stat.traceback), stat.size) for stat in top_stats[:10]]
    
    def _check_memory_freed(self, container_id: str) -> None:
        """Check if memory was freed after position close."""
        if container_id not in self.container_refs:
            return
            
        container = self.container_refs[container_id]()
        if not container:
            return
        
        # Get memory before and after a short delay
        before = self._get_container_memory_usage(container)
        gc.collect()  # Force garbage collection
        after = self._get_container_memory_usage(container)
        
        if after >= before * 0.95:  # Memory didn't drop by at least 5%
            logger.warning(f"Memory not freed after position close in {container_id}: {after} bytes")
            if self.emit_warnings:
                self._emit_memory_warning(container_id, "Memory not freed after position close")
    
    def _emit_memory_warning(self, container_id: str, message: str) -> None:
        """Emit memory warning event."""
        # Rate limit warnings
        now = datetime.now()
        if self._last_warning_time and (now - self._last_warning_time).seconds < 60:
            return
        
        self._last_warning_time = now
        
        warning_event = Event(
            event_type="MEMORY_WARNING",
            payload={
                'container_id': container_id,
                'message': message,
                'memory_by_type': dict(self.memory_by_type),
                'event_count': self.event_counts.get(container_id, 0)
            },
            source_id=self.__class__.__name__,
            container_id=container_id
        )
        
        logger.warning(f"Memory warning for {container_id}: {message}")
    
    def _emit_memory_snapshot_event(self, snapshot: MemorySnapshot) -> None:
        """Emit memory snapshot as event."""
        event = Event(
            event_type="MEMORY_SNAPSHOT",
            payload={
                'container_id': snapshot.container_id,
                'memory_mb': snapshot.memory_bytes / 1024 / 1024,
                'event_count': snapshot.event_count,
                'tracer_events': snapshot.tracer_events,
                'open_positions': snapshot.open_positions,
                'gc_stats': snapshot.gc_stats
            },
            source_id=self.__class__.__name__,
            container_id=snapshot.container_id
        )
        
        # Would need event bus reference to actually publish
        logger.debug(f"Memory snapshot for {snapshot.container_id}: {snapshot.memory_bytes / 1024 / 1024:.2f}MB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        total_events = sum(self.event_counts.values())
        
        # Latest snapshots by container
        latest_by_container = {}
        for snapshot in self.memory_snapshots:
            if snapshot.container_id not in latest_by_container or \
               snapshot.timestamp > latest_by_container[snapshot.container_id].timestamp:
                latest_by_container[snapshot.container_id] = snapshot
        
        return {
            'total_events_processed': total_events,
            'containers_monitored': len(self.container_refs),
            'memory_by_event_type': dict(self.memory_by_type),
            'latest_snapshots': {
                cid: {
                    'memory_mb': snap.memory_bytes / 1024 / 1024,
                    'event_count': snap.event_count,
                    'open_positions': snap.open_positions
                }
                for cid, snap in latest_by_container.items()
            },
            'total_snapshots': len(self.memory_snapshots)
        }


# ============================================
# Integration Example
# ============================================
"""
How to use the memory observer with your containers:
"""

class Container:
    def __init__(self, config):
        self.container_id = config['id']
        self.event_bus = EventBus(self.container_id)
        
        # Setup observers
        for observer_config in config.get('observers', []):
            if observer_config['type'] == 'metrics':
                # Your existing metrics
                observer = MetricsEventTracer(observer_config)
                
            elif observer_config['type'] == 'memory':
                # Memory monitoring
                observer = ContainerMemoryObserver(
                    sampling_rate=observer_config.get('sampling_rate', 0.01),
                    deep_inspection_interval=observer_config.get('inspection_interval', 1000),
                    emit_warnings=observer_config.get('emit_warnings', True),
                    memory_threshold_mb=observer_config.get('threshold_mb', 100.0)
                )
                observer.attach_to_container(self)  # Pass container reference!
                
            self.event_bus.attach_observer(observer)


# YAML Configuration:
"""
containers:
  - id: portfolio_1
    observers:
      - type: metrics
        retention_policy: trade_complete
        
      - type: memory
        sampling_rate: 0.01      # Sample 1% of events
        inspection_interval: 1000 # Deep inspection every 1000 events
        emit_warnings: true
        threshold_mb: 50.0       # Warn if container uses >50MB
        track_allocations: false  # Enable for detailed memory profiling
"""


# ============================================
# Benefits of Memory Observer
# ============================================
"""
1. **Memory Leak Detection**: Identifies when memory isn't freed after trades close
2. **Performance Profiling**: See which event types consume most memory
3. **Early Warning System**: Get alerts before OOM errors
4. **Optimization Insights**: Data-driven memory optimization
5. **Production Monitoring**: Lightweight enough for production use

The memory observer turns your event system into a self-monitoring platform,
providing granular diagnostics on system performance through the same
event-driven architecture!
"""


# ============================================
# Mock Implementation for Testing
# ============================================

@dataclass
class MockMetricsCalculator:
    """Mock calculator for testing."""
    
    update_from_trade_called: bool = False
    update_portfolio_value_called: bool = False
    last_trade_data: Optional[Dict[str, Any]] = None
    
    def update_from_trade(self, entry_price: float, exit_price: float, 
                         quantity: float, direction: str) -> None:
        """Track that method was called."""
        self.update_from_trade_called = True
        self.last_trade_data = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'direction': direction
        }
    
    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Track that method was called."""
        self.update_portfolio_value_called = True
    
    def get_metrics(self) -> Dict[str, float]:
        """Return mock metrics."""
        return {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.10
        }
