# Standardized Communication Architecture Plan

## Executive Summary

This document outlines a comprehensive plan to standardize communication across the ADMF-PC system while addressing the current sub-container routing issues. The plan implements a hybrid approach where:

1. **System-wide Event Router** handles all cross-container communication
2. **Sub-container Communication** uses simplified patterns within container boundaries
3. **Automatic Registration** ensures all containers (including dynamically created ones) participate in routing

## Current State Analysis

### Issues Identified

1. **Sub-container Registration Gap**: Dynamically created sub-containers (e.g., individual strategy containers within StrategyContainer) are not registered with the Event Router
2. **Inconsistent Communication Patterns**: Mix of Event Router and old patterns creates confusion
3. **Event Scope Confusion**: DOWNWARD scope doesn't work for broad data distribution like BAR events
4. **Missing Event Router Setup**: Some containers declare routing config but aren't registered with router

### Current Working Patterns

1. **Cross-sibling Communication**: RiskContainer ↔ ExecutionContainer via Event Router (ORDER/FILL events)
2. **Hierarchical Data Flow**: DataContainer → descendants via old pattern (BAR events)
3. **Indicator Distribution**: IndicatorContainer finds and delivers to specific containers

## Proposed Architecture

### 1. Two-Tier Communication Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM-WIDE COMMUNICATION                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Event Router (Tier 1)                    │   │
│  │  • Cross-container communication                        │   │
│  │  • Siblings (Risk ↔ Execution)                         │   │
│  │  • Parent-child when needed                            │   │
│  │  • Global broadcasts                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Container Boundaries (Tier 2)                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │         Within Container Hierarchy              │   │   │
│  │  │  • Parent → Child: Direct event bus             │   │   │
│  │  │  • Child → Parent: Direct event bus             │   │   │
│  │  │  • Optimized for performance                    │   │   │
│  │  │  • Simple patterns for internal communication    │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Event Flow Classification

#### Tier 1: System-Wide Event Router
- **Cross-sibling**: Risk ↔ Execution (ORDER/FILL)
- **Cross-hierarchy**: Strategy → Portfolio → Risk (SIGNAL)
- **Global services**: IndicatorContainer → All subscribers (INDICATOR)
- **System events**: End-of-backtest, configuration changes

#### Tier 2: Container Hierarchy
- **Data streaming**: Data → All descendants (BAR)
- **Internal coordination**: Parent ↔ Child within same container
- **Sub-container management**: StrategyContainer ↔ sub-strategies

### 3. Automatic Registration System

```python
class ContainerEventInterface:
    """Enhanced interface with automatic sub-container registration"""
    
    def __init__(self):
        self._event_router: Optional[EventRouterProtocol] = None
        self._sub_containers: List[ComposableContainerProtocol] = []
        self._publications: List[EventPublication] = []
        self._subscriptions: List[EventSubscription] = []
    
    def register_with_router(self, router: EventRouterProtocol) -> None:
        """Register with router and cascade to sub-containers"""
        self._event_router = router
        
        # Register self
        if self._publications:
            router.register_publisher(self.metadata.container_id, self._publications)
        
        if self._subscriptions:
            router.register_subscriber(
                self.metadata.container_id,
                self._subscriptions,
                self.handle_routed_event
            )
        
        # CASCADE: Register all sub-containers automatically
        for sub_container in self._sub_containers:
            if hasattr(sub_container, 'register_with_router'):
                sub_container.register_with_router(router)
    
    def add_child_container(self, child: ComposableContainerProtocol) -> None:
        """Override to auto-register new children with router"""
        super().add_child_container(child)
        
        # Auto-register with router if we're already registered
        if self._event_router and hasattr(child, 'register_with_router'):
            child.register_with_router(self._event_router)
```

## Implementation Plan

### Phase 1: Foundation (Immediate)

#### 1.1 Update Base Container Implementation
- Enhance `ContainerEventInterface` with automatic sub-container registration
- Add router cascade registration to `add_child_container()`
- Implement consistent event routing setup

#### 1.2 Fix Current Event Router Issues
- Ensure all main containers register with Event Router during initialization
- Fix StrategyContainer sub-container registration gap
- Implement proper router setup in container bootstrap

```python
# In StrategyContainer._initialize_multi_strategy()
async def _initialize_multi_strategy(self) -> None:
    """Initialize multiple strategies as sub-containers with router registration."""
    # ... existing code ...
    
    for i, strategy_config in enumerate(self.strategies_config):
        # Create sub-container
        sub_container = StrategyContainer(
            config=sub_container_config,
            container_id=f"{self.metadata.container_id}_{strategy_name}"
        )
        
        # Add as child (this will auto-register with router)
        self.add_child_container(sub_container)
        
        # Ensure sub-container has router access for SIGNAL publishing
        if self._event_router:
            sub_container.register_with_router(self._event_router)
```

#### 1.3 Standardize Event Publishing Patterns
- All SIGNAL events use Event Router (EventScope.PARENT)
- All ORDER/FILL events use Event Router (EventScope.SIBLINGS)
- All BAR/data events use hierarchical pattern (target_scope="children")

### Phase 2: Event Flow Standardization (Next)

#### 2.1 Implement Event Flow Classification
```yaml
# Standard event flow configuration
event_flows:
  system_wide:  # Tier 1: Event Router
    - events: ["ORDER", "FILL"]
      pattern: "siblings"
      scope: "SIBLINGS"
    
    - events: ["SIGNAL"]
      pattern: "hierarchy_cross"
      scope: "PARENT"
    
    - events: ["INDICATOR"]
      pattern: "targeted_delivery"
      scope: "GLOBAL"
  
  container_internal:  # Tier 2: Direct event bus
    - events: ["BAR", "TICK"]
      pattern: "hierarchical_broadcast"
      scope: "children"
    
    - events: ["PORTFOLIO_UPDATE"]
      pattern: "parent_child"
      scope: "parent"
```

#### 2.2 Update Container Event Declarations
```python
# Example: RiskContainer with standardized declarations
class RiskContainer(BaseComposableContainer, ContainerEventInterface):
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure Event Router patterns (Tier 1)
        self._configure_router_events(config)
        super().__init__(...)
    
    def _configure_router_events(self, config: Dict[str, Any]) -> None:
        """Configure Event Router communication"""
        config['events'] = config.get('events', {})
        
        # System-wide publications
        config['events']['publishes'] = [
            {
                'events': ['ORDER'],
                'scope': 'SIBLINGS',  # To ExecutionContainer
                'tier': 'system_wide'
            }
        ]
        
        # System-wide subscriptions
        config['events']['subscribes'] = [
            {
                'events': ['FILL'],
                'scope': 'SIBLINGS',  # From ExecutionContainer
                'tier': 'system_wide'
            }
        ]
```

### Phase 3: Advanced Features (Future)

#### 3.1 Router Performance Optimization
- Event batching for high-frequency events
- Subscription filtering
- Dead letter queue for failed deliveries

#### 3.2 Container Communication Debugging
- Event flow visualization
- Communication pattern analysis
- Performance monitoring

#### 3.3 Configuration-Driven Routing
- YAML-based routing topology
- Dynamic subscription management
- Runtime routing changes

## Detailed Implementation

### Event Router Registration Fix

```python
# In execution/containers.py
class StrategyContainer(BaseComposableContainer, ContainerEventInterface):
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Initialize Event Router interface FIRST
        ContainerEventInterface.__init__(self)
        
        # Configure Event Router patterns
        self._configure_event_routing(config)
        
        # Then initialize base container
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        
        # Container-specific initialization
        self.strategy: Optional[Strategy] = None
        self.signal_aggregator = None
        
    def _configure_event_routing(self, config: Dict[str, Any]) -> None:
        """Configure Event Router communication patterns"""
        if 'events' not in config:
            config['events'] = {}
        
        # Declare what we publish via Event Router
        router_publications = [
            EventPublication(
                events={EventType.SIGNAL},
                scope=EventScope.PARENT,  # To PortfolioContainer
                qos=QoS.AT_LEAST_ONCE
            )
        ]
        self.declare_publications(router_publications)
        
        # Declare what we subscribe to via Event Router  
        router_subscriptions = [
            EventSubscription(
                source="indicator_container",
                events={EventType.INDICATORS},
                filters={"subscriber": self.metadata.container_id}
            )
        ]
        self.declare_subscriptions(router_subscriptions)
    
    async def _emit_signals(self, signals: List, timestamp, market_data: Dict[str, Any]) -> None:
        """Emit signals using Event Router"""
        if signals:
            signal_event = Event(
                event_type=EventType.SIGNAL,
                payload={
                    'timestamp': timestamp,
                    'signals': signals,
                    'market_data': market_data,
                    'source': self.metadata.container_id
                },
                timestamp=timestamp
            )
            
            # Use Event Router for system-wide communication
            logger.info(f"🚀 StrategyContainer publishing SIGNAL via Event Router")
            self.publish_routed_event(signal_event, EventScope.PARENT)
```

### Bootstrap Registration Integration

```python
# In core/containers/bootstrap.py
async def setup_container_communication(root_container: ComposableContainerProtocol) -> EventRouterProtocol:
    """Set up Event Router and register all containers"""
    from ..events.routing.router import EventRouter
    
    # Create Event Router
    router = EventRouter()
    
    # Register all containers in hierarchy with router
    def register_container_tree(container):
        if hasattr(container, 'register_with_router'):
            container.register_with_router(router)
            logger.info(f"Registered {container.metadata.name} with Event Router")
        
        # Recursively register children
        for child in getattr(container, 'child_containers', []):
            register_container_tree(child)
    
    register_container_tree(root_container)
    
    logger.info(f"Event Router setup complete - {router.get_stats()}")
    return router
```

## Migration Strategy

### Step 1: Immediate Fixes (This Session)
1. Fix sub-container registration in StrategyContainer
2. Ensure Event Router is properly initialized and containers registered
3. Standardize SIGNAL event publishing to use Event Router

### Step 2: Systematic Review (Next Session)
1. Audit all container event declarations
2. Classify events into Tier 1 (Event Router) vs Tier 2 (Hierarchical)
3. Update container implementations for consistency

### Step 3: Testing and Validation
1. Test multi-strategy signal flow end-to-end
2. Verify Event Router performance and reliability
3. Add comprehensive event flow debugging

## Expected Outcomes

### Immediate Benefits
- Sub-containers can successfully publish SIGNAL events
- Consistent communication patterns across system
- Clear separation between system-wide and internal communication

### Long-term Benefits
- Easy to add new container types
- Predictable event routing behavior
- Strong debugging and monitoring capabilities
- Configuration-driven communication topology

## Success Metrics

1. **Signal Flow Success Rate**: 100% of generated signals reach RiskContainer
2. **Event Router Registration**: All containers (including sub-containers) registered
3. **Communication Consistency**: All similar communication patterns use same approach
4. **Performance**: No degradation in event processing speed
5. **Debuggability**: Clear visibility into event flow paths

---

This plan provides a clear path forward for standardizing communication while maintaining the architectural benefits of both Event Router for cross-container communication and simplified patterns for intra-container coordination.