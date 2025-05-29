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

from .types import (
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
    EventBus,
    ContainerEventBus
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
    
    # Event bus implementations
    "EventBus",
    "ContainerEventBus",
    
    # Subscription management
    "Subscription",
    "SubscriptionManager",
    "WeakSubscriptionManager",
    "create_subscription_manager",
    
    # Isolation management
    "EventIsolationManager",
    "IsolatedEventPublisher",
    "validate_event_isolation",
    "get_isolation_manager"
]