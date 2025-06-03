"""Base Communication Adapter Interface.

This module provides the abstract base class for all communication adapters
in the ADMF-PC system. Communication adapters handle the translation between
internal event formats and external communication mechanisms.

Key responsibilities:
- Event format translation
- Protocol abstraction
- Error handling and recovery
- Metrics tracking
- Correlation ID propagation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, Set, Callable
import time
import uuid

from ..logging.container_logger import ContainerLogger
from ..events.types import Event, EventType


@dataclass
class AdapterMetrics:
    """Metrics tracked by communication adapters."""
    
    events_sent: int = 0
    events_received: int = 0
    events_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors_count: int = 0
    last_error_time: Optional[datetime] = None
    total_latency_ms: float = 0.0
    connection_attempts: int = 0
    connection_failures: int = 0
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        total_events = self.events_sent + self.events_received
        if total_events == 0:
            return 0.0
        return self.total_latency_ms / total_events
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total_events = self.events_sent + self.events_received
        if total_events == 0:
            return 0.0
        return self.events_failed / total_events


@dataclass
class AdapterConfig:
    """Configuration for communication adapters."""
    
    name: str
    adapter_type: str
    retry_attempts: int = 3
    retry_delay_ms: int = 100
    timeout_ms: int = 5000
    buffer_size: int = 1000
    enable_compression: bool = False
    enable_encryption: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class MessageHandler(Protocol):
    """Protocol for handling received messages."""
    
    def __call__(self, message: Any, correlation_id: Optional[str] = None) -> None:
        """Handle a received message."""
        ...


class CommunicationAdapter(ABC):
    """Abstract base class for all communication adapters.
    
    This class provides the foundation for implementing various communication
    protocols (WebSocket, gRPC, ZeroMQ, etc.) in a consistent manner.
    """
    
    def __init__(self, config: AdapterConfig, logger: Optional[ContainerLogger] = None):
        """Initialize the communication adapter.
        
        Args:
            config: Adapter configuration
            logger: Container logger for structured logging
        """
        self.config = config
        self.logger = logger or ContainerLogger(
            container_id=f"comm_adapter_{config.name}",
            component_name=f"adapter.{config.adapter_type}",
            log_level="INFO"
        )
        
        self.metrics = AdapterMetrics()
        self._is_connected = False
        self._is_running = False
        self._message_handlers: Set[MessageHandler] = set()
        self._setup_complete = False
        
        # Log adapter initialization
        self.logger.info(
            "Initializing communication adapter",
            adapter_type=config.adapter_type,
            adapter_name=config.name,
            config=config
        )
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the communication channel.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the communication channel."""
        pass
    
    @abstractmethod
    async def send_raw(self, data: bytes, correlation_id: Optional[str] = None) -> bool:
        """Send raw bytes through the communication channel.
        
        Args:
            data: Raw bytes to send
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            True if send successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def receive_raw(self) -> Optional[bytes]:
        """Receive raw bytes from the communication channel.
        
        Returns:
            Received bytes or None if no data available
        """
        pass
    
    # Lifecycle methods
    
    async def setup(self) -> None:
        """Setup the adapter (called once before first use)."""
        if self._setup_complete:
            self.logger.warning("Setup called multiple times")
            return
            
        self.logger.info("Setting up communication adapter")
        
        try:
            # Connect to the communication channel
            connected = await self.connect()
            if not connected:
                raise RuntimeError("Failed to establish connection")
            
            self._is_connected = True
            self._is_running = True
            self._setup_complete = True
            
            self.logger.info(
                "Communication adapter setup complete",
                is_connected=self._is_connected
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to setup communication adapter",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def cleanup(self) -> None:
        """Cleanup the adapter (called once when shutting down)."""
        if not self._setup_complete:
            return
            
        self.logger.info("Cleaning up communication adapter")
        
        try:
            self._is_running = False
            
            # Disconnect from communication channel
            if self._is_connected:
                await self.disconnect()
                self._is_connected = False
            
            # Clear message handlers
            self._message_handlers.clear()
            
            # Log final metrics
            self.logger.info(
                "Communication adapter cleanup complete",
                metrics=self.metrics
            )
            
            self._setup_complete = False
            
        except Exception as e:
            self.logger.error(
                "Error during communication adapter cleanup",
                error=str(e),
                error_type=type(e).__name__
            )
    
    # High-level API methods
    
    async def send_event(self, event: Event) -> bool:
        """Send an event through the communication channel.
        
        Args:
            event: Event to send
            
        Returns:
            True if send successful, False otherwise
        """
        if not self._is_connected:
            self.logger.warning(
                "Attempted to send event while disconnected",
                event_type=event.event_type
            )
            return False
        
        start_time = time.time()
        correlation_id = event.metadata.get('correlation_id', str(uuid.uuid4()))
        
        try:
            # Serialize event to bytes (subclasses can override)
            data = await self.serialize_event(event)
            
            # Send through adapter
            success = await self.send_raw(data, correlation_id)
            
            if success:
                self.metrics.events_sent += 1
                self.metrics.bytes_sent += len(data)
            else:
                self.metrics.events_failed += 1
            
            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms
            
            # Log send result
            self.logger.debug(
                "Event sent through adapter",
                event_type=event.event_type,
                correlation_id=correlation_id,
                success=success,
                latency_ms=latency_ms,
                bytes_sent=len(data)
            )
            
            return success
            
        except Exception as e:
            self.metrics.errors_count += 1
            self.metrics.last_error_time = datetime.now()
            
            self.logger.error(
                "Failed to send event",
                event_type=event.event_type,
                correlation_id=correlation_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            return False
    
    async def process_incoming(self) -> None:
        """Process incoming messages from the communication channel."""
        if not self._is_connected or not self._is_running:
            return
        
        try:
            # Receive raw data
            data = await self.receive_raw()
            if data is None:
                return
            
            start_time = time.time()
            self.metrics.events_received += 1
            self.metrics.bytes_received += len(data)
            
            # Deserialize to event (subclasses can override)
            event, correlation_id = await self.deserialize_event(data)
            
            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms
            
            # Notify handlers
            for handler in self._message_handlers:
                try:
                    handler(event, correlation_id)
                except Exception as e:
                    self.logger.error(
                        "Message handler error",
                        handler=handler,
                        error=str(e),
                        error_type=type(e).__name__
                    )
            
            self.logger.debug(
                "Processed incoming message",
                event_type=event.event_type,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                bytes_received=len(data)
            )
            
        except Exception as e:
            self.metrics.errors_count += 1
            self.metrics.last_error_time = datetime.now()
            
            self.logger.error(
                "Failed to process incoming message",
                error=str(e),
                error_type=type(e).__name__
            )
    
    # Handler management
    
    def add_handler(self, handler: MessageHandler) -> None:
        """Add a message handler.
        
        Args:
            handler: Handler to add
        """
        self._message_handlers.add(handler)
        self.logger.debug(
            "Added message handler",
            handler=handler,
            total_handlers=len(self._message_handlers)
        )
    
    def remove_handler(self, handler: MessageHandler) -> None:
        """Remove a message handler.
        
        Args:
            handler: Handler to remove
        """
        self._message_handlers.discard(handler)
        self.logger.debug(
            "Removed message handler",
            handler=handler,
            total_handlers=len(self._message_handlers)
        )
    
    # Serialization methods (can be overridden by subclasses)
    
    async def serialize_event(self, event: Event) -> bytes:
        """Serialize an event to bytes.
        
        Default implementation uses JSON. Subclasses can override for
        different serialization formats.
        
        Args:
            event: Event to serialize
            
        Returns:
            Serialized bytes
        """
        import json
        
        # Convert event to dict
        event_dict = {
            'event_type': event.event_type.name if isinstance(event.event_type, EventType) else event.event_type,
            'payload': event.payload,
            'timestamp': event.timestamp.isoformat(),
            'source_id': event.source_id,
            'container_id': event.container_id,
            'metadata': event.metadata
        }
        
        # Add correlation ID if present in metadata
        if 'correlation_id' in event.metadata:
            event_dict['correlation_id'] = event.metadata['correlation_id']
        
        # Serialize to JSON bytes
        return json.dumps(event_dict).encode('utf-8')
    
    async def deserialize_event(self, data: bytes) -> tuple[Event, Optional[str]]:
        """Deserialize bytes to an event.
        
        Default implementation expects JSON. Subclasses can override for
        different serialization formats.
        
        Args:
            data: Bytes to deserialize
            
        Returns:
            Tuple of (event, correlation_id)
        """
        import json
        from datetime import datetime
        
        # Parse JSON
        event_dict = json.loads(data.decode('utf-8'))
        
        # Extract correlation ID
        correlation_id = event_dict.get('correlation_id')
        
        # Parse event type
        event_type_str = event_dict['event_type']
        try:
            event_type = EventType[event_type_str] if event_type_str in EventType.__members__ else event_type_str
        except:
            event_type = event_type_str
        
        # Create Event object
        event = Event(
            event_type=event_type,
            payload=event_dict.get('payload', {}),
            timestamp=datetime.fromisoformat(event_dict['timestamp']),
            source_id=event_dict.get('source_id'),
            container_id=event_dict.get('container_id'),
            metadata=event_dict.get('metadata', {})
        )
        
        # Store correlation ID in metadata if present
        if correlation_id:
            event.metadata['correlation_id'] = correlation_id
        
        return event, correlation_id
    
    # Utility methods
    
    def get_metrics(self) -> AdapterMetrics:
        """Get current adapter metrics.
        
        Returns:
            Current metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset adapter metrics."""
        self.metrics = AdapterMetrics()
        self.logger.info("Reset adapter metrics")
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._is_connected
    
    @property
    def is_running(self) -> bool:
        """Check if adapter is running."""
        return self._is_running