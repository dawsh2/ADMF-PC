"""
Event system for ADMF-PC using Protocol + Composition.

This module provides a clean, composable event infrastructure where:
- EventBus is pure pub/sub with no tracing logic
- Tracing is added via observers using composition
- Everything uses protocols, no inheritance
- Integrates seamlessly with container architecture

All existing imports work exactly the same - 100% compatibility maintained.
"""

from .protocols import (
    EventBusProtocol,
    EventHandler,
    EventObserverProtocol,
    EventTracerProtocol,
    EventStorageProtocol,
    EventFilterProtocol,
    MetricsCalculatorProtocol,
    ResultExtractor,
)

from .bus import EventBus

from .types import (
    Event,
    EventType,
    create_market_event,
    create_signal_event,
    create_system_event,
    create_error_event,
    create_classification_event,
    # Time utilities
    parse_event_time,
    event_age,
    is_event_stale,
    format_event_time,
    get_event_window,
)

from .filters import (
    # Filter functions
    strategy_filter,
    container_filter,
    order_filter,
    order_ownership_filter,
    portfolio_symbol_filter,
    combine_filters,
    any_of_filters,
    symbol_filter,
    metadata_filter,
)

from .tracing import (
    # Core tracing
    EventTracer,
    MetricsObserver,
    create_tracer_from_config,
    
    # Storage
    MemoryEventStorage,
    DiskEventStorage,
    HierarchicalEventStorage,
    create_storage_backend,
    
    # Analysis features
    MetricsExtractor,
    ObjectiveFunctionExtractor,
    EventQueryInterface,
    EventFlowValidator,
    
    # Signal storage & replay
    SignalIndex,
    ClassifierChangeIndex,
    MultiSymbolSignal,
    SignalStorageManager,
    SignalStorageProtocol,
    ClassifierStateProtocol,
)

# from .isolation import EventIsolationManager  # Removed - using shared root bus
from .subscriptions import WeakSubscriptionManager
from .configuration import (
    TraceLevel,
    TraceConfiguration,
    get_trace_config,
    apply_trace_level,
)
from .tracer_setup import (
    setup_multi_strategy_tracer,
    finalize_multi_strategy_tracer,
)

# Create aliases for backward compatibility
SubscriptionManager = WeakSubscriptionManager

# Data mining moved to src.analytics.mining

__all__ = [
    # Protocols
    'EventBusProtocol',
    'EventHandler',
    'EventObserverProtocol',
    'EventTracerProtocol',
    'EventStorageProtocol',
    'EventFilterProtocol',
    'MetricsCalculatorProtocol',
    'ResultExtractor',
    
    # Core
    'EventBus',
    'Event',
    'EventType',
    
    # Event creation
    'create_market_event',
    'create_signal_event',
    'create_system_event',
    'create_error_event',
    'create_classification_event',
    
    # Time utilities
    'parse_event_time',
    'event_age',
    'is_event_stale',
    'format_event_time',
    'get_event_window',
    
    # Tracing
    'EventTracer',
    'MetricsObserver',
    'create_tracer_from_config',
    
    # Storage
    'MemoryEventStorage',
    'DiskEventStorage',
    'HierarchicalEventStorage',
    'create_storage_backend',
    
    # Filters
    'strategy_filter',
    'container_filter',
    'order_filter',
    'order_ownership_filter',
    'portfolio_symbol_filter',
    'combine_filters',
    'any_of_filters',
    'symbol_filter',
    'metadata_filter',
    
    # Advanced features
    'WeakSubscriptionManager',
    'SubscriptionManager',
    'TraceLevel',
    'TraceConfiguration',
    'EventQueryInterface',
    # 'EventIsolationManager',  # Removed - using shared root bus
    'MetricsExtractor',
    'ObjectiveFunctionExtractor',
    'EventFlowValidator',
    'get_trace_config',
    'apply_trace_level',
    
    # Signal storage & replay
    'SignalIndex',
    'ClassifierChangeIndex',
    'MultiSymbolSignal',
    'SignalStorageManager',
    'SignalStorageProtocol',
    'ClassifierStateProtocol',
    
    # Tracer setup
    'setup_multi_strategy_tracer',
    'finalize_multi_strategy_tracer',
    
]