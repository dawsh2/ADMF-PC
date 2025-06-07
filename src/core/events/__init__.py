"""
Containerized event system for ADMF-PC.

This package provides the event-driven communication infrastructure
with complete isolation between containers. Each container (backtest,
optimization trial, etc.) gets its own event bus, preventing any
cross-contamination during parallel execution.

Key Components:
- Event: Core event structure with container isolation
- EventBus: Container-specific event routing
- SubscriptionManager: Lifecycle-aware subscription management
- EventIsolationManager: Ensures complete separation between containers

Example Usage:
    ```python
    # In container setup
    isolation_manager = get_isolation_manager()
    event_bus = isolation_manager.create_container_bus("backtest_001")
    
    # In component initialization
    subscription_manager = SubscriptionManager(event_bus, "my_strategy")
    subscription_manager.subscribe(EventType.BAR, self.on_bar)
    
    # Publishing events
    event = create_market_event(
        EventType.BAR,
        symbol="AAPL",
        timestamp=datetime.now(),
        data={"close": 150.0},
        container_id="backtest_001"
    )
    event_bus.publish(event)
    
    # Cleanup
    subscription_manager.unsubscribe_all()
    isolation_manager.remove_container_bus("backtest_001")
    ```
"""

from ..types.events import (
    Event,
    EventType,
    EventHandler,
    EventPublisher,
    EventSubscriber,
    EventBusProtocol,
    EventCapable,
    create_market_event,
    create_signal_event,
    create_system_event,
    create_error_event
)

from .event_bus import (
    EventBus
)

from .subscription_manager import (
    Subscription,
    SubscriptionManager,
    WeakSubscriptionManager,
    create_subscription_manager
)

from .isolation import (
    ContainerProtocol,
    EventIsolationManager,
    IsolatedEventPublisher,
    validate_event_isolation,
    get_isolation_manager
)

# Semantic events and type flow analysis
from .semantic import (
    SemanticEvent,
    EventCategory,
    MarketDataEvent,
    FeatureEvent,
    TradingSignal,
    OrderEvent,
    FillEvent,
    PortfolioUpdateEvent,
    create_event_with_context,
    create_caused_event,
    validate_semantic_event,
    feature_to_signal,
    signal_to_order,
    order_to_fill
)

from .type_flow_analysis import (
    FlowNode,
    TypeTransition,
    ValidationResult,
    EventTypeRegistry,
    TypeFlowAnalyzer,
    ContainerTypeInferencer
)

from .type_flow_integration import (
    TypeFlowValidator,
    create_default_validator,
    validate_route_network,
    get_semantic_event_suggestions,
    create_type_flow_report
)

from .type_flow_visualization import (
    TypeFlowVisualizer,
    create_flow_visualization,
    validate_and_visualize,
    export_mermaid_diagram
)

# Result extraction framework
from .result_extraction import (
    ResultExtractor,
    PortfolioMetricsExtractor,
    SignalExtractor,
    FillExtractor,
    OrderExtractor,
    RiskDecisionExtractor,
    RegimeChangeExtractor,
    PerformanceSnapshotExtractor
)

from .extractor_registry import (
    ExtractorRegistry,
    ExtractorConfig
)

from .enhanced_tracer import (
    EnhancedEventTracer,
    StreamingResultProcessor
)



__all__ = [
    # Types and protocols
    "Event",
    "EventType",
    "EventHandler",
    "EventPublisher",
    "EventSubscriber",
    "EventBusProtocol",
    "EventCapable",
    "ContainerProtocol",
    
    # Event creation helpers
    "create_market_event",
    "create_signal_event",
    "create_system_event",
    "create_error_event",
    
    # Event bus implementation
    "EventBus",
    
    # Subscription management
    "Subscription",
    "SubscriptionManager",
    "WeakSubscriptionManager",
    "create_subscription_manager",
    
    # Isolation management
    "EventIsolationManager",
    "IsolatedEventPublisher",
    "validate_event_isolation",
    "get_isolation_manager",
    
    # Semantic events
    "SemanticEvent",
    "EventCategory",
    "MarketDataEvent",
    "FeatureEvent",
    "TradingSignal",
    "OrderEvent",
    "FillEvent",
    "PortfolioUpdateEvent",
    "create_event_with_context",
    "create_caused_event",
    "validate_semantic_event",
    "feature_to_signal",
    "signal_to_order",
    "order_to_fill",
    
    # Type flow analysis
    "FlowNode",
    "TypeTransition",
    "ValidationResult",
    "EventTypeRegistry",
    "TypeFlowAnalyzer",
    "ContainerTypeInferencer",
    
    # Type flow integration
    "TypeFlowValidator",
    "create_default_validator",
    "validate_route_network",
    "get_semantic_event_suggestions",
    "create_type_flow_report",
    
    # Type flow visualization
    "TypeFlowVisualizer",
    "create_flow_visualization",
    "validate_and_visualize",
    "export_mermaid_diagram",
    
    # Result extraction framework
    "ResultExtractor",
    "PortfolioMetricsExtractor",
    "SignalExtractor",
    "FillExtractor",
    "OrderExtractor",
    "RiskDecisionExtractor",
    "RegimeChangeExtractor",
    "PerformanceSnapshotExtractor",
    
    # Extractor registry
    "ExtractorRegistry",
    "ExtractorConfig",
    
    # Enhanced event tracing
    "EnhancedEventTracer",
    "StreamingResultProcessor"
]