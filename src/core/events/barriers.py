"""
Unified barrier system for event synchronization.

This module consolidates ALL synchronization components using a simple, unified protocol.
All barriers follow Protocol+Composition with no inheritance.

Barriers handle:
- Data alignment across symbols/timeframes
- Order duplicate prevention  
- Timing constraints (rate limits, trading hours)
- Event coordination

The event bus handles the actual synchronization - barriers just provide logic.
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import logging

from .types import Event, EventType

logger = logging.getLogger(__name__)


# === MONITORING AND STATS ===

@dataclass
class BarrierStats:
    """Statistics for monitoring barrier performance."""
    events_passed: int = 0
    events_blocked: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        total = self.events_passed + self.events_blocked
        return self.events_passed / total if total > 0 else 0.0
    
    @property
    def total_events(self) -> int:
        """Total events processed."""
        return self.events_passed + self.events_blocked
    
    def reset(self) -> None:
        """Reset statistics."""
        self.events_passed = 0
        self.events_blocked = 0
        self.last_reset = datetime.now()


# === TIME PROVIDER PROTOCOL ===

@runtime_checkable
class TimeProviderProtocol(Protocol):
    """Protocol for providing time to barriers - supports both real-time and simulated time."""
    
    def get_current_time(self, event: Optional[Event] = None) -> datetime:
        """Get current time - either from event (simulated) or system clock (live)."""
        ...


class EventTimeProvider:
    """Time provider that uses event timestamps for backtesting."""
    
    def get_current_time(self, event: Optional[Event] = None) -> datetime:
        """Get time from event timestamp if available, otherwise use system time."""
        if event and hasattr(event, 'timestamp') and event.timestamp:
            return event.timestamp
        return datetime.now()


class SystemTimeProvider:
    """Time provider that always uses system clock for live trading."""
    
    def get_current_time(self, event: Optional[Event] = None) -> datetime:
        """Always return system time regardless of event."""
        return datetime.now()


# === UNIFIED BARRIER PROTOCOL ===

@runtime_checkable
class BarrierProtocol(Protocol):
    """Unified protocol for all barriers - simple and clean."""
    
    def should_proceed(self, event: Event) -> bool:
        """Check if processing should proceed based on current state."""
        ...
    
    def update_state(self, event: Event) -> None:
        """Update barrier state from incoming event."""
        ...
    
    def reset(self) -> None:
        """Reset barrier state."""
        ...


# === ENUMS AND TYPES ===

class AlignmentMode(Enum):
    """How to handle data alignment."""
    ALL = 'all'          # Wait for all required data
    ANY = 'any'          # Process when any data arrives
    BEST_EFFORT = 'best_effort'  # Use whatever is available
    ROLLING = 'rolling'  # Rolling window alignment


class TimeframeAlignment(Enum):
    """How to align different timeframes."""
    STRICT = 'strict'    # Exact timestamp match
    NEAREST = 'nearest'  # Closest available bar
    FORWARD_FILL = 'forward_fill'  # Use last known value


class BarrierMode(Enum):
    """How barriers are combined."""
    ALL = 'all'      # All barriers must pass
    ANY = 'any'      # Any barrier passing is sufficient
    NONE = 'none'    # No barriers (pass through)


@dataclass
class DataRequirement:
    """Specification for required data."""
    symbols: List[str]
    timeframes: List[str] = field(default_factory=lambda: ['1m'])
    min_history: int = 1
    alignment_mode: AlignmentMode = AlignmentMode.ALL
    timeframe_alignment: TimeframeAlignment = TimeframeAlignment.NEAREST
    timeout_ms: Optional[int] = None
    
    @property
    def required_keys(self) -> List[Tuple[str, str]]:
        """Get required symbol/timeframe combinations."""
        return [(s, t) for s in self.symbols for t in self.timeframes]
    
    def to_key(self) -> Tuple:
        """Create hashable key for this requirement."""
        return (
            tuple(sorted(self.symbols)),
            tuple(sorted(self.timeframes)), 
            self.alignment_mode.value,
            self.min_history
        )


@dataclass
class BarBuffer:
    """Simple buffer for storing bars."""
    bars: List[Event] = field(default_factory=list)
    max_size: int = 1000
    
    def add_bar(self, bar_event: Event) -> None:
        """Add bar to buffer."""
        self.bars.append(bar_event)
        if len(self.bars) > self.max_size:
            self.bars.pop(0)
    
    def get_history(self, count: int) -> List[Event]:
        """Get last N bars."""
        return self.bars[-count:] if count <= len(self.bars) else self.bars.copy()
    
    def get_latest(self) -> Optional[Event]:
        """Get most recent bar."""
        return self.bars[-1] if self.bars else None


# === BARRIER IMPLEMENTATIONS ===

@dataclass
class DataAlignmentBarrier:
    """
    Barrier for data alignment across symbols/timeframes.
    Much simpler than the old TimeAlignmentBuffer.
    """
    required_symbols: List[str]
    required_timeframes: List[str] = field(default_factory=lambda: ['1m'])
    timeout_seconds: Optional[float] = None
    time_provider: TimeProviderProtocol = field(default_factory=EventTimeProvider)
    
    def __post_init__(self):
        self._received_data: Dict[str, datetime] = {}
        self._last_reset = self.time_provider.get_current_time()
    
    def should_proceed(self, event: Event) -> bool:
        """Check if we have all required data."""
        if event.event_type != EventType.BAR.value:
            return True  # Non-bar events pass through
        
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe')
        
        if symbol in self.required_symbols and timeframe in self.required_timeframes:
            key = f"{symbol}_{timeframe}"
            self._received_data[key] = self.time_provider.get_current_time(event)
        
        # Check if we have all required combinations
        required_keys = {
            f"{symbol}_{tf}" 
            for symbol in self.required_symbols 
            for tf in self.required_timeframes
        }
        
        return required_keys.issubset(self._received_data.keys())
    
    def update_state(self, event: Event) -> None:
        """State updated in should_proceed."""
        pass
    
    def reset(self) -> None:
        """Reset received data."""
        self._received_data.clear()
        self._last_reset = self.time_provider.get_current_time()


@dataclass
class OrderStateBarrier:
    """
    Barrier for preventing duplicate orders.
    Much simpler than the old OrderTracker + DuplicateOrderPrevention.
    """
    container_id: str
    prevent_duplicates: bool = True
    
    def __post_init__(self):
        self._pending_orders: Dict[str, set] = defaultdict(set)  # symbol -> {order_ids}
    
    def should_proceed(self, event: Event) -> bool:
        """Check if signal should be processed (no pending orders)."""
        if not self.prevent_duplicates:
            return True
            
        if event.event_type == EventType.SIGNAL.value:
            symbol = event.payload.get('symbol')
            if symbol and symbol in self._pending_orders:
                return len(self._pending_orders[symbol]) == 0
        
        return True
    
    def update_state(self, event: Event) -> None:
        """Track order state changes."""
        if event.event_type == EventType.ORDER.value:
            symbol = event.payload.get('symbol')
            order_id = event.payload.get('order_id')
            if symbol and order_id:
                self._pending_orders[symbol].add(order_id)
                
        elif event.event_type in (EventType.FILL.value, EventType.CANCEL.value):
            symbol = event.payload.get('symbol')
            order_id = event.payload.get('order_id')
            if symbol and order_id:
                self._pending_orders[symbol].discard(order_id)
    
    def reset(self) -> None:
        """Clear all pending orders."""
        self._pending_orders.clear()


@dataclass
class TimingBarrier:
    """
    Barrier for timing constraints (rate limits, trading hours, etc.).
    """
    max_signals_per_minute: Optional[int] = None
    trading_hours: Optional[tuple] = None  # (start_hour, end_hour)
    min_time_between_signals: Optional[float] = None  # seconds
    time_provider: TimeProviderProtocol = field(default_factory=EventTimeProvider)
    
    def __post_init__(self):
        self._signal_times: List[datetime] = []
        self._last_signal_time: Optional[datetime] = None
    
    def should_proceed(self, event: Event) -> bool:
        """Check timing constraints."""
        if event.event_type != EventType.SIGNAL.value:
            return True
            
        current_time = self.time_provider.get_current_time(event)
        
        # Check trading hours
        if self.trading_hours:
            start_hour, end_hour = self.trading_hours
            if not (start_hour <= current_time.hour < end_hour):
                return False
        
        # Check minimum time between signals
        if self.min_time_between_signals and self._last_signal_time:
            time_diff = (current_time - self._last_signal_time).total_seconds()
            if time_diff < self.min_time_between_signals:
                return False
        
        # Check rate limit
        if self.max_signals_per_minute:
            # Remove signals older than 1 minute
            cutoff = current_time.timestamp() - 60
            self._signal_times = [t for t in self._signal_times if t.timestamp() > cutoff]
            
            if len(self._signal_times) >= self.max_signals_per_minute:
                return False
        
        return True
    
    def update_state(self, event: Event) -> None:
        """Update timing state."""
        if event.event_type == EventType.SIGNAL.value:
            current_time = self.time_provider.get_current_time(event)
            self._signal_times.append(current_time)
            self._last_signal_time = current_time
    
    def reset(self) -> None:
        """Reset timing state."""
        self._signal_times.clear()
        self._last_signal_time = None


@dataclass
class CompositeBarrier:
    """
    Combines multiple barriers with AND/OR logic.
    This is the main barrier system containers use.
    """
    barriers: List[BarrierProtocol]
    mode: BarrierMode = BarrierMode.ALL
    
    def __post_init__(self):
        """Initialize stats tracking."""
        self.stats = BarrierStats()
    
    def should_proceed(self, event: Event) -> bool:
        """Check all barriers according to mode."""
        if not self.barriers:
            result = True
        else:
            results = [barrier.should_proceed(event) for barrier in self.barriers]
            
            if self.mode == BarrierMode.ALL:
                result = all(results)
            elif self.mode == BarrierMode.ANY:
                result = any(results)
            else:  # NONE
                result = True
        
        # Track statistics
        if result:
            self.stats.events_passed += 1
        else:
            self.stats.events_blocked += 1
            
        return result
    
    def update_state(self, event: Event) -> None:
        """Update all barriers."""
        for barrier in self.barriers:
            barrier.update_state(event)
    
    def reset(self) -> None:
        """Reset all barriers."""
        for barrier in self.barriers:
            barrier.reset()
    
    def add_barrier(self, barrier: BarrierProtocol) -> None:
        """Add a barrier to the composite."""
        self.barriers.append(barrier)
    
    def remove_barrier(self, barrier: BarrierProtocol) -> None:
        """Remove a barrier from the composite."""
        if barrier in self.barriers:
            self.barriers.remove(barrier)
    
    def get_stats(self) -> Dict[str, BarrierStats]:
        """Get statistics for each barrier."""
        stats_dict = {
            'composite': self.stats
        }
        
        # Add individual barrier stats if they support it
        for i, barrier in enumerate(self.barriers):
            if hasattr(barrier, 'stats'):
                stats_dict[f"barrier_{i}"] = barrier.stats
            else:
                # Create basic stats for barriers without tracking
                stats_dict[f"barrier_{i}"] = BarrierStats()
        
        return stats_dict


# === FACTORY FUNCTIONS ===

def create_data_barrier(symbols: List[str], timeframes: List[str] = None, time_provider: TimeProviderProtocol = None) -> DataAlignmentBarrier:
    """Create a data alignment barrier for specific symbols/timeframes."""
    return DataAlignmentBarrier(
        required_symbols=symbols,
        required_timeframes=timeframes or ['1m'],
        time_provider=time_provider or EventTimeProvider()
    )


def create_order_barrier(container_id: str, prevent_duplicates: bool = True) -> OrderStateBarrier:
    """Create an order state barrier for duplicate prevention."""
    return OrderStateBarrier(
        container_id=container_id,
        prevent_duplicates=prevent_duplicates
    )


def create_timing_barrier(
    max_per_minute: Optional[int] = None,
    trading_hours: Optional[tuple] = None,
    min_gap_seconds: Optional[float] = None,
    time_provider: TimeProviderProtocol = None
) -> TimingBarrier:
    """Create a timing constraint barrier."""
    return TimingBarrier(
        max_signals_per_minute=max_per_minute,
        trading_hours=trading_hours,
        min_time_between_signals=min_gap_seconds,
        time_provider=time_provider or EventTimeProvider()
    )


def create_standard_barriers(
    container_id: str,
    symbols: List[str],
    timeframes: List[str] = None,
    prevent_duplicates: bool = True,
    max_signals_per_minute: Optional[int] = None,
    time_provider: TimeProviderProtocol = None
) -> CompositeBarrier:
    """Create the standard barrier setup most containers need."""
    barriers = []
    
    # Use EventTimeProvider by default for backtesting compatibility
    provider = time_provider or EventTimeProvider()
    
    # Data alignment
    if symbols:
        barriers.append(create_data_barrier(symbols, timeframes, provider))
    
    # Order state
    if prevent_duplicates:
        barriers.append(create_order_barrier(container_id, True))
    
    # Rate limiting
    if max_signals_per_minute:
        barriers.append(create_timing_barrier(max_per_minute=max_signals_per_minute, time_provider=provider))
    
    return CompositeBarrier(barriers, BarrierMode.ALL)


def setup_barriers_from_config(config: Dict[str, Any], time_provider: TimeProviderProtocol = None) -> CompositeBarrier:
    """
    Create barriers from configuration.
    
    Example config:
    {
        'data_alignment': {
            'symbols': ['AAPL', 'GOOGL'],
            'timeframes': ['1m', '5m']
        },
        'order_control': {
            'prevent_duplicates': True,
            'container_id': 'portfolio_1'
        },
        'timing': {
            'max_signals_per_minute': 10,
            'trading_hours': [9, 16],
            'min_gap_seconds': 5.0
        },
        'time_mode': 'event'  # 'event' for backtesting, 'system' for live trading
    }
    """
    barriers = []
    
    # Determine time provider based on config or parameter
    if time_provider is None:
        time_mode = config.get('time_mode', 'event')  # Default to event time for backtesting
        if time_mode == 'system':
            time_provider = SystemTimeProvider()
        else:
            time_provider = EventTimeProvider()
    
    # Data alignment barrier
    if 'data_alignment' in config:
        data_config = config['data_alignment']
        barrier = create_data_barrier(
            symbols=data_config.get('symbols', []),
            timeframes=data_config.get('timeframes', ['1m']),
            time_provider=time_provider
        )
        barriers.append(barrier)
    
    # Order control barrier
    if 'order_control' in config:
        order_config = config['order_control']
        barrier = create_order_barrier(
            container_id=order_config.get('container_id', 'unknown'),
            prevent_duplicates=order_config.get('prevent_duplicates', True)
        )
        barriers.append(barrier)
    
    # Timing barrier
    if 'timing' in config:
        timing_config = config['timing']
        barrier = create_timing_barrier(
            max_per_minute=timing_config.get('max_signals_per_minute'),
            trading_hours=tuple(timing_config['trading_hours']) if 'trading_hours' in timing_config else None,
            min_gap_seconds=timing_config.get('min_gap_seconds'),
            time_provider=time_provider
        )
        barriers.append(barrier)
    
    return CompositeBarrier(barriers, BarrierMode.ALL)


# === CONTAINER INTEGRATION ===

class BarrierMixin:
    """
    Mixin for containers to add barrier functionality.
    Much simpler than the old StrategyOrchestrator.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._barriers: Optional[CompositeBarrier] = None
    
    def set_barriers(self, barriers: CompositeBarrier) -> None:
        """Set the barrier system for this container."""
        self._barriers = barriers
    
    def should_process_event(self, event: Event) -> bool:
        """Check if event should be processed through barriers."""
        if not self._barriers:
            return True
        return self._barriers.should_proceed(event)
    
    def update_barrier_state(self, event: Event) -> None:
        """Update barrier state."""
        if self._barriers:
            self._barriers.update_state(event)
    
    def reset_barriers(self) -> None:
        """Reset all barriers."""
        if self._barriers:
            self._barriers.reset()


# === EXPORTS ===

__all__ = [
    # Time providers
    'TimeProviderProtocol',
    'EventTimeProvider', 
    'SystemTimeProvider',
    
    # Barrier protocol and stats
    'BarrierProtocol',
    'BarrierStats',
    
    # Barrier implementations
    'DataAlignmentBarrier',
    'OrderStateBarrier', 
    'TimingBarrier',
    'CompositeBarrier',
    
    # Factory functions
    'create_data_barrier',
    'create_order_barrier',
    'create_timing_barrier', 
    'create_standard_barriers',
    'setup_barriers_from_config',
    
    # Container integration
    'BarrierMixin',
    
    # Enums and types
    'AlignmentMode',
    'TimeframeAlignment',
    'BarrierMode',
    'DataRequirement',
    'BarBuffer',
]