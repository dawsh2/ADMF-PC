"""Event types and creation helpers."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from datetime import datetime, time, timezone
from enum import Enum
import uuid


class EventType(Enum):
    """Standard event types in the system."""
    # Market data events
    BAR = "BAR"
    TICK = "TICK"
    
    # Feature events
    FEATURES = "FEATURES"
    
    # Trading events
    SIGNAL = "SIGNAL"
    ORDER_REQUEST = "ORDER_REQUEST"
    ORDER = "ORDER"
    FILL = "FILL"
    
    # Portfolio events
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    POSITION_UPDATE = "POSITION_UPDATE"
    POSITION_OPEN = "POSITION_OPEN"
    POSITION_CLOSE = "POSITION_CLOSE"
    
    # System events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    ERROR = "ERROR"
    
    # Analysis events
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_BREACH = "RISK_BREACH"
    CLASSIFICATION = "CLASSIFICATION"  # Classifier outputs


@dataclass
class Event:
    """
    Enhanced event structure with tracing capabilities.
    
    Attributes:
        event_type: Type of event
        payload: Event data
        source_id: ID of component that created event
        container_id: ID of container that owns the source
        correlation_id: ID to correlate related events
        causation_id: ID of event that caused this event
        timestamp: When event was created
        metadata: Additional event metadata
        event_id: Unique event identifier (auto-generated)
        sequence_number: Sequence number within correlation
        target_container: Container this event is targeted at
    """
    event_type: str
    payload: Dict[str, Any]
    source_id: Optional[str] = None
    container_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    sequence_number: Optional[int] = None
    target_container: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type,
            'payload': self.payload,
            'source_id': self.source_id,
            'container_id': self.container_id,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
            'event_id': self.event_id,
            'sequence_number': self.sequence_number,
            'target_container': self.target_container
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            event_type=data['event_type'],
            payload=data['payload'],
            source_id=data.get('source_id'),
            container_id=data.get('container_id'),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            timestamp=timestamp or datetime.now(),
            metadata=data.get('metadata', {}),
            event_id=data.get('event_id', f"evt_{uuid.uuid4().hex[:12]}"),
            sequence_number=data.get('sequence_number'),
            target_container=data.get('target_container')
        )


# Event creation helpers

def create_market_event(
    event_type: EventType,
    symbol: str,
    data: Dict[str, Any],
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a market data event."""
    return Event(
        event_type=event_type.value,
        payload={
            'symbol': symbol,
            **data
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'market_data'}
    )

def create_signal_event(
    symbol: str,
    direction: str,
    strength: float,
    strategy_id: str,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None,
    combo_id: Optional[str] = None
) -> Event:
    """Create a trading signal event."""
    return Event(
        event_type=EventType.SIGNAL.value,
        payload={
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'strategy_id': strategy_id,
            'combo_id': combo_id
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'trading'}
    )

def create_system_event(
    event_type: EventType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a system event."""
    return Event(
        event_type=event_type.value,
        payload={
            'message': message,
            'details': details or {}
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'system'}
    )

def create_error_event(
    error_type: str,
    error_message: str,
    original_event: Optional[Event] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create an error event."""
    payload = {
        'error_type': error_type,
        'error_message': error_message
    }
    
    if original_event:
        payload['original_event_type'] = original_event.event_type
        payload['original_event_id'] = original_event.metadata.get('event_id')
    
    return Event(
        event_type=EventType.ERROR.value,
        payload=payload,
        source_id=source_id,
        container_id=container_id,
        causation_id=original_event.metadata.get('event_id') if original_event else None,
        metadata={'category': 'error'}
    )

def create_classification_event(
    symbol: str,
    regime: str,
    confidence: float,
    classifier_id: str,
    previous_regime: Optional[str] = None,
    features: Optional[Dict[str, float]] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """Create a classification event."""
    return Event(
        event_type=EventType.CLASSIFICATION.value,
        payload={
            'symbol': symbol,
            'regime': regime,
            'confidence': confidence,
            'classifier_id': classifier_id,
            'previous_regime': previous_regime,
            'features': features or {},
            'is_regime_change': previous_regime is not None and previous_regime != regime
        },
        source_id=source_id,
        container_id=container_id,
        metadata={'category': 'classification'}
    )


# ============================================
# Time utilities for event timestamps
# ============================================

def parse_event_time(time_str: Union[str, datetime]) -> datetime:
    """
    Parse various time formats into datetime.
    
    Args:
        time_str: Time string in ISO format or datetime object
        
    Returns:
        Parsed datetime object
    """
    if isinstance(time_str, datetime):
        return time_str
    
    # Try common formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    # Try ISO format parsing
    try:
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except:
        raise ValueError(f"Could not parse time string: {time_str}")


def event_age(event: Event, now: Optional[datetime] = None) -> float:
    """
    Calculate age of event in seconds.
    
    Args:
        event: Event to check
        now: Current time (default: datetime.now())
        
    Returns:
        Age in seconds
    """
    if now is None:
        now = datetime.now()
    
    # Ensure both times have same timezone awareness
    if event.timestamp.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    elif event.timestamp.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=event.timestamp.tzinfo)
    
    return (now - event.timestamp).total_seconds()


def is_event_stale(event: Event, max_age_seconds: float = 60.0) -> bool:
    """
    Check if event is older than max age.
    
    Args:
        event: Event to check
        max_age_seconds: Maximum age in seconds
        
    Returns:
        True if event is stale
    """
    return event_age(event) > max_age_seconds


def format_event_time(dt: datetime, include_micros: bool = False) -> str:
    """
    Format datetime for event display.
    
    Args:
        dt: Datetime to format
        include_micros: Include microseconds
        
    Returns:
        Formatted time string
    """
    if include_micros:
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # milliseconds
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def get_event_window(events: list[Event], 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> list[Event]:
    """
    Filter events by time window.
    
    Args:
        events: List of events
        start_time: Window start (inclusive)
        end_time: Window end (inclusive)
        
    Returns:
        Events within time window
    """
    filtered = events
    
    if start_time:
        filtered = [e for e in filtered if e.timestamp >= start_time]
    
    if end_time:
        filtered = [e for e in filtered if e.timestamp <= end_time]
    
    return filtered