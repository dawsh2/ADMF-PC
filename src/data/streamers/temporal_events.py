"""
Temporal event emission for data-driven constraint management.

This module provides TemporalEventEmitter that emits natural data boundary events
like END_OF_DAY and END_OF_STREAM, which portfolio can respond to for position management.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, time
import logging

from ...core.events import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class MarketHours:
    """Define market hours for temporal boundary detection."""
    market_open: time = time(9, 30)   # 9:30 AM
    market_close: time = time(16, 0)  # 4:00 PM  
    timezone: str = "US/Eastern"
    
    def is_market_close_time(self, dt: datetime) -> bool:
        """Check if datetime represents market close."""
        return dt.time() >= self.market_close
    
    def is_market_open_time(self, dt: datetime) -> bool:
        """Check if datetime represents market open."""
        return dt.time() >= self.market_open


@dataclass
class TemporalEventEmitter:
    """
    Emits time-based constraint events from data stream.
    
    This component watches the data stream and emits natural boundary events:
    - END_OF_DAY: When trading day ends
    - END_OF_STREAM: When backtest data is exhausted
    - MARKET_HOLIDAY: When market is closed (future enhancement)
    
    Portfolio components can subscribe to these events to manage positions.
    """
    
    # Configuration
    market_hours: MarketHours = field(default_factory=MarketHours)
    emit_end_of_day: bool = True
    emit_end_of_stream: bool = True
    
    # State tracking
    last_bar_time: Optional[datetime] = None
    last_trading_day: Optional[datetime] = None
    total_bars_processed: int = 0
    
    # Event bus for publishing
    event_bus: Optional[Any] = None
    
    # Callbacks for custom event handling
    event_callbacks: List[Callable[[Event], None]] = field(default_factory=list)
    
    def initialize(self, event_bus: Any) -> None:
        """Initialize with event bus for publishing events."""
        self.event_bus = event_bus
        logger.info("TemporalEventEmitter initialized")
    
    def register_callback(self, callback: Callable[[Event], None]) -> None:
        """Register callback for temporal events."""
        self.event_callbacks.append(callback)
    
    def process_bar(self, bar_event: Event) -> None:
        """
        Process a bar event and check for temporal constraints.
        
        Call this method for each bar event in your data stream.
        """
        # Extract bar timestamp
        bar_timestamp = self._extract_bar_timestamp(bar_event)
        if not bar_timestamp:
            return
        
        self.total_bars_processed += 1
        
        # Check for end of day
        if self.emit_end_of_day and self._is_end_of_trading_day(bar_timestamp):
            self._emit_end_of_day_event(bar_event, bar_timestamp)
        
        # Update tracking
        self.last_bar_time = bar_timestamp
        self.last_trading_day = bar_timestamp.date()
    
    def process_end_of_stream(self, final_bar_event: Optional[Event] = None) -> None:
        """
        Signal that the data stream has ended.
        
        Call this when your data streamer has no more data.
        """
        if self.emit_end_of_stream:
            self._emit_end_of_stream_event(final_bar_event)
    
    def _is_end_of_trading_day(self, bar_timestamp: datetime) -> bool:
        """Check if this bar represents end of trading day."""
        # Simple check: if this bar is at or after market close
        if self.market_hours.is_market_close_time(bar_timestamp):
            return True
        
        # Could also check if next bar would be on different day
        # This would require lookahead, so keeping simple for now
        return False
    
    def _is_last_bar_in_dataset(self, bar_timestamp: datetime) -> bool:
        """Check if this is the last bar in the dataset."""
        # This would be called by the data streamer when it knows
        # there are no more bars. For now, we handle this via 
        # explicit process_end_of_stream() call
        return False
    
    def _emit_end_of_day_event(self, bar_event: Event, bar_timestamp: datetime) -> None:
        """Emit END_OF_DAY event."""
        symbol = bar_event.payload.get('symbol', 'UNKNOWN')
        
        eod_event = Event(
            event_type='END_OF_DAY',
            payload={
                'symbol': symbol,
                'date': bar_timestamp.date().isoformat(),
                'last_bar_time': bar_timestamp.isoformat(),
                'market_close_time': self.market_hours.market_close.isoformat(),
                'bars_processed_today': 1  # Could track this more precisely
            },
            source_id='temporal_event_emitter',
            metadata={'category': 'temporal_constraint'}
        )
        
        self._publish_event(eod_event)
        logger.debug(f"Emitted END_OF_DAY for {symbol} at {bar_timestamp}")
    
    def _emit_end_of_stream_event(self, final_bar_event: Optional[Event] = None) -> None:
        """Emit END_OF_STREAM event."""
        payload = {
            'reason': 'backtest_complete',
            'total_bars_processed': self.total_bars_processed,
            'final_timestamp': self.last_bar_time.isoformat() if self.last_bar_time else None,
            'final_trading_day': self.last_trading_day.isoformat() if self.last_trading_day else None
        }
        
        if final_bar_event:
            payload['final_symbol'] = final_bar_event.payload.get('symbol', 'UNKNOWN')
            
        eos_event = Event(
            event_type='END_OF_STREAM',
            payload=payload,
            source_id='temporal_event_emitter',
            metadata={'category': 'temporal_constraint'}
        )
        
        self._publish_event(eos_event)
        logger.info(f"Emitted END_OF_STREAM after processing {self.total_bars_processed} bars")
    
    def _publish_event(self, event: Event) -> None:
        """Publish event via event bus and callbacks."""
        # Publish via event bus if available
        if self.event_bus and hasattr(self.event_bus, 'publish'):
            self.event_bus.publish(event)
        
        # Call registered callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in temporal event callback: {e}")
    
    def _extract_bar_timestamp(self, bar_event: Event) -> Optional[datetime]:
        """Extract timestamp from bar event."""
        # Try different possible timestamp fields
        timestamp_fields = [
            'bar_close_time',
            'timestamp', 
            'bar_timestamp',
            'close_time'
        ]
        
        for field in timestamp_fields:
            timestamp = bar_event.payload.get(field)
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        return datetime.fromisoformat(timestamp)
                    except ValueError:
                        continue
                elif isinstance(timestamp, datetime):
                    return timestamp
        
        # Try extracting from nested bar data
        bar_data = bar_event.payload.get('bar')
        if bar_data and hasattr(bar_data, 'timestamp'):
            return bar_data.timestamp
        
        logger.warning(f"Could not extract timestamp from bar event: {bar_event.payload.keys()}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get temporal event emitter statistics."""
        return {
            'total_bars_processed': self.total_bars_processed,
            'last_bar_time': self.last_bar_time.isoformat() if self.last_bar_time else None,
            'last_trading_day': self.last_trading_day.isoformat() if self.last_trading_day else None,
            'callbacks_registered': len(self.event_callbacks),
            'emit_end_of_day': self.emit_end_of_day,
            'emit_end_of_stream': self.emit_end_of_stream
        }


# Convenience functions for common usage patterns

def create_temporal_emitter(event_bus: Any, 
                          market_hours: Optional[MarketHours] = None) -> TemporalEventEmitter:
    """Create and initialize a temporal event emitter."""
    emitter = TemporalEventEmitter(
        market_hours=market_hours or MarketHours()
    )
    emitter.initialize(event_bus)
    return emitter


def add_temporal_events_to_streamer(streamer: Any, event_bus: Any) -> TemporalEventEmitter:
    """
    Add temporal event emission to an existing data streamer.
    
    This function creates a TemporalEventEmitter and sets up callbacks
    so that the streamer automatically emits temporal events.
    """
    emitter = create_temporal_emitter(event_bus)
    
    # If streamer has an event emission method, hook into it
    if hasattr(streamer, 'on_bar_emitted'):
        streamer.on_bar_emitted = lambda event: emitter.process_bar(event)
    
    # If streamer has an end-of-data method, hook into it  
    if hasattr(streamer, 'on_stream_complete'):
        streamer.on_stream_complete = lambda: emitter.process_end_of_stream()
    
    return emitter