"""Communication Module for ADMF-PC.

This module provides the communication infrastructure for the ADMF-PC system,
enabling various communication protocols and patterns through a unified adapter
interface.

Key Components:
- CommunicationAdapter: Abstract base class for all adapters
- AdapterConfig: Configuration for adapters
- AdapterMetrics: Metrics tracking for adapters

The communication module supports:
- Multiple protocols (WebSocket, gRPC, ZeroMQ, etc.)
- Event serialization/deserialization
- Correlation ID tracking
- Comprehensive metrics
- Error handling and recovery
- Full logging integration

Example usage:
    from src.core.communication import CommunicationAdapter, AdapterConfig
    
    # Configure adapter
    config = AdapterConfig(
        name="market_data",
        adapter_type="websocket",
        retry_attempts=3,
        timeout_ms=5000
    )
    
    # Create adapter (using concrete implementation)
    adapter = WebSocketAdapter(config)
    
    # Setup and use
    await adapter.setup()
    await adapter.send_event(event)
    await adapter.cleanup()
"""

from .base_adapter import (
    CommunicationAdapter,
    AdapterConfig,
    AdapterMetrics,
    MessageHandler
)
from .pipeline_adapter import (
    PipelineCommunicationAdapter,
    EventTransformer,
    PipelineStage
)
from .factory import (
    EventCommunicationFactory,
    CommunicationLayer
)

__all__ = [
    "CommunicationAdapter",
    "AdapterConfig",
    "AdapterMetrics",
    "MessageHandler",
    "PipelineCommunicationAdapter",
    "EventTransformer",
    "PipelineStage",
    "EventCommunicationFactory",
    "CommunicationLayer"
]

# Version info
__version__ = "1.0.0"