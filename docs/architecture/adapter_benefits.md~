# Event Communication + Logging System Integration

## Core Integration Principle

Event communication adapters must integrate seamlessly with your container-aware logging system to maintain observability while respecting the same architectural boundaries and lifecycle management.

## Event Scope Alignment

Your logging system already classifies events by scope - the communication adapters should use the same classification:

```python
class EventScope(Enum):
    INTERNAL_BUS = "internal_bus"               # Within container - adapter shouldn't touch
    EXTERNAL_FAST = "external_fast_tier"       # Cross-container fast - Pipeline adapter
    EXTERNAL_STANDARD = "external_standard_tier" # Cross-container standard - Broadcast adapter  
    EXTERNAL_RELIABLE = "external_reliable_tier" # Cross-container reliable - Selective adapter
    COMPONENT_INTERNAL = "component_internal"   # Within component - adapter shouldn't touch
    LIFECYCLE_MANAGEMENT = "lifecycle_management" # Log management operations
```

## Logging-Aware Communication Adapters

### Enhanced Pipeline Adapter with Logging Integration

```python
class LoggingAwarePipelineCommunicationAdapter:
    """Pipeline adapter that integrates with container logging system"""
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        # Create adapter-specific logger
        self.logger = ContainerLogger(
            coordinator_id, 
            "pipeline_adapter",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        # Track adapter metrics
        self.metrics = {
            'events_routed': 0,
            'routing_errors': 0,
            'total_latency_ms': 0,
            'transformations': 0
        }
    
    def setup_flow(self, containers: List[Container]) -> None:
        """Set up pipeline flow with comprehensive logging"""
        
        self.logger.log_info(
            "Setting up pipeline communication flow",
            container_count=len(containers),
            container_ids=[c.container_id for c in containers],
            event_flow="external_standard_tier",
            lifecycle_operation="adapter_setup"
        )
        
        for i, container in enumerate(containers[:-1]):
            next_container = containers[i + 1]
            
            # Set up transformation mapping with logging
            container.on_output_event(
                lambda event, target=next_container: self.transform_and_forward_with_logging(event, target)
            )
            
            self.logger.log_debug(
                "Pipeline stage configured",
                source_container=container.container_id,
                target_container=next_container.container_id,
                stage_number=i,
                event_flow="external_standard_tier"
            )
    
    def transform_and_forward_with_logging(self, event: Event, target: Container) -> None:
        """Transform and forward with comprehensive event tracking"""
        start_time = time.time()
        
        try:
            # Set correlation ID for cross-container tracking
            correlation_id = event.get_correlation_id() or f"pipeline_{uuid.uuid4().hex[:8]}"
            event.set_correlation_id(correlation_id)
            
            # Log the routing decision
            self.logger.log_debug(
                "Routing event through pipeline",
                event_id=event.id,
                event_type=event.type,
                source_container=event.source_container_id,
                target_container=target.container_id,
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            
            # Transform event based on type mapping
            transformed_event = self.event_transformer.transform(event)
            
            # Forward to target with event tracking
            target.receive_event(transformed_event)
            
            # Update metrics and log success
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['events_routed'] += 1
            self.metrics['total_latency_ms'] += latency_ms
            self.metrics['transformations'] += 1
            
            self.logger.log_debug(
                "Pipeline routing completed",
                event_id=event.id,
                correlation_id=correlation_id,
                latency_ms=round(latency_ms, 2),
                success=True,
                event_flow="external_standard_tier"
            )
            
        except Exception as e:
            # Log routing failures
            self.metrics['routing_errors'] += 1
            
            self.logger.log_error(
                "Pipeline routing failed", 
                event_id=event.id,
                source_container=event.source_container_id,
                target_container=target.container_id,
                error=str(e),
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            
            # Re-raise to maintain error handling semantics
            raise
    
    def get_adapter_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics for monitoring"""
        avg_latency = (
            self.metrics['total_latency_ms'] / max(self.metrics['events_routed'], 1)
        )
        
        return {
            "adapter_type": "pipeline",
            "events_routed": self.metrics['events_routed'],
            "routing_errors": self.metrics['routing_errors'],
            "error_rate": self.metrics['routing_errors'] / max(self.metrics['events_routed'], 1),
            "average_latency_ms": round(avg_latency, 2),
            "transformations": self.metrics['transformations']
        }
```

### Enhanced Hierarchical Adapter with Logging

```python
class LoggingAwareHierarchicalCommunicationAdapter:
    """Hierarchical adapter for classifier-first with detailed logging"""
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        # Create adapter logger
        self.logger = ContainerLogger(
            coordinator_id,
            "hierarchical_adapter", 
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        # Track parent-child relationships for logging
        self.hierarchy_map: Dict[str, List[str]] = {}
        
    def setup_hierarchy(self, parent: Container, children: List[Container]) -> None:
        """Set up hierarchy with regime context logging"""
        
        child_ids = [c.container_id for c in children]
        self.hierarchy_map[parent.container_id] = child_ids
        
        self.logger.log_info(
            "Setting up hierarchical communication",
            parent_container=parent.container_id,
            child_containers=child_ids,
            child_count=len(children),
            event_flow="external_standard_tier",
            lifecycle_operation="adapter_setup"
        )
        
        # Parent events flow down to all children
        parent.on_context_event(
            lambda event: self.broadcast_to_children_with_logging(event, children, parent.container_id)
        )
        
        # Children events aggregate up to parent
        for child in children:
            child.on_result_event(
                lambda event, child_id=child.container_id: self.aggregate_to_parent_with_logging(
                    event, parent, child_id
                )
            )
    
    def broadcast_to_children_with_logging(self, event: Event, children: List[Container], parent_id: str) -> None:
        """Broadcast context to children with detailed logging"""
        
        correlation_id = event.get_correlation_id() or f"hierarchy_{uuid.uuid4().hex[:8]}"
        event.set_correlation_id(correlation_id)
        
        self.logger.log_info(
            "Broadcasting context event to children",
            event_type=event.type,
            parent_container=parent_id,
            child_count=len(children),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
        
        success_count = 0
        for child in children:
            try:
                child.receive_context_event(event)
                success_count += 1
                
                self.logger.log_debug(
                    "Context event delivered to child",
                    child_container=child.container_id,
                    event_type=event.type,
                    correlation_id=correlation_id,
                    event_flow="external_standard_tier"
                )
                
            except Exception as e:
                self.logger.log_error(
                    "Failed to deliver context to child",
                    child_container=child.container_id,
                    parent_container=parent_id,
                    event_type=event.type,
                    error=str(e),
                    correlation_id=correlation_id,
                    event_flow="external_standard_tier"
                )
        
        self.logger.log_info(
            "Context broadcast completed",
            successful_deliveries=success_count,
            total_children=len(children),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
    
    def aggregate_to_parent_with_logging(self, event: Event, parent: Container, child_id: str) -> None:
        """Aggregate child results with performance tracking"""
        
        correlation_id = event.get_correlation_id()
        
        self.logger.log_debug(
            "Aggregating child result to parent",
            child_container=child_id,
            parent_container=parent.container_id,
            event_type=event.type,
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
        
        try:
            parent.receive_child_result(event)
            
            self.logger.log_debug(
                "Child result aggregated successfully",
                child_container=child_id,
                parent_container=parent.container_id,
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            
        except Exception as e:
            self.logger.log_error(
                "Failed to aggregate child result",
                child_container=child_id,
                parent_container=parent.container_id,
                event_type=event.type,
                error=str(e),
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            raise
```

### Enhanced Selective Adapter for Complex Routing

```python
class LoggingAwareSelectiveCommunicationAdapter:
    """Selective routing with rule evaluation logging"""
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        self.logger = ContainerLogger(
            coordinator_id,
            "selective_adapter",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        self.routing_rules: List[Tuple[Callable, Container, str]] = []
        self.rule_metrics: Dict[str, Dict[str, int]] = {}
        
    def add_routing_rule(self, condition: Callable, target: Container, rule_name: str = None) -> None:
        """Add routing rule with logging setup"""
        
        rule_name = rule_name or f"rule_{len(self.routing_rules)}"
        self.routing_rules.append((condition, target, rule_name))
        
        # Initialize metrics for this rule
        self.rule_metrics[rule_name] = {
            'evaluations': 0,
            'matches': 0,
            'routes': 0,
            'errors': 0
        }
        
        self.logger.log_info(
            "Added routing rule",
            rule_name=rule_name,
            target_container=target.container_id,
            rule_count=len(self.routing_rules),
            lifecycle_operation="adapter_setup"
        )
    
    def setup_selective_routing(self, source: Container) -> None:
        """Set up content-based routing with comprehensive logging"""
        
        self.logger.log_info(
            "Setting up selective routing",
            source_container=source.container_id,
            rule_count=len(self.routing_rules),
            event_flow="external_reliable_tier",  # Selective routing typically for important decisions
            lifecycle_operation="adapter_setup"
        )
        
        source.on_event(
            lambda event: self.route_event_with_logging(event, source.container_id)
        )
    
    def route_event_with_logging(self, event: Event, source_container_id: str) -> None:
        """Route event with detailed rule evaluation logging"""
        
        correlation_id = event.get_correlation_id() or f"selective_{uuid.uuid4().hex[:8]}"
        event.set_correlation_id(correlation_id)
        
        self.logger.log_debug(
            "Starting selective routing evaluation",
            event_id=event.id,
            event_type=event.type,
            source_container=source_container_id,
            rule_count=len(self.routing_rules),
            correlation_id=correlation_id,
            event_flow="external_reliable_tier"
        )
        
        routed = False
        evaluation_results = []
        
        for condition, target, rule_name in self.routing_rules:
            try:
                # Update rule metrics
                self.rule_metrics[rule_name]['evaluations'] += 1
                
                # Evaluate rule condition
                start_time = time.time()
                matches = condition(event)
                evaluation_time_ms = (time.time() - start_time) * 1000
                
                evaluation_results.append({
                    'rule_name': rule_name,
                    'matches': matches,
                    'evaluation_time_ms': round(evaluation_time_ms, 2),
                    'target_container': target.container_id
                })
                
                self.logger.log_debug(
                    "Rule evaluation completed",
                    rule_name=rule_name,
                    matches=matches,
                    evaluation_time_ms=round(evaluation_time_ms, 2),
                    target_container=target.container_id,
                    correlation_id=correlation_id,
                    event_flow="external_reliable_tier"
                )
                
                if matches:
                    self.rule_metrics[rule_name]['matches'] += 1
                    
                    # Route the event
                    target.receive_event(event)
                    routed = True
                    
                    self.rule_metrics[rule_name]['routes'] += 1
                    
                    self.logger.log_info(
                        "Event routed by selective adapter",
                        rule_name=rule_name,
                        event_id=event.id,
                        source_container=source_container_id,
                        target_container=target.container_id,
                        correlation_id=correlation_id,
                        event_flow="external_reliable_tier"
                    )
                    
                    break  # First matching rule wins
                    
            except Exception as e:
                self.rule_metrics[rule_name]['errors'] += 1
                
                self.logger.log_error(
                    "Error evaluating routing rule",
                    rule_name=rule_name,
                    event_id=event.id,
                    source_container=source_container_id,
                    error=str(e),
                    correlation_id=correlation_id,
                    event_flow="external_reliable_tier"
                )
        
        # Log final routing result
        if not routed:
            self.logger.log_warning(
                "No routing rule matched event",
                event_id=event.id,
                event_type=event.type,
                source_container=source_container_id,
                rules_evaluated=len(self.routing_rules),
                correlation_id=correlation_id,
                event_flow="external_reliable_tier"
            )
        
        # Log evaluation summary
        self.logger.log_debug(
            "Selective routing completed",
            event_id=event.id,
            routed=routed,
            rules_evaluated=len(evaluation_results),
            evaluation_results=evaluation_results,
            correlation_id=correlation_id,
            event_flow="external_reliable_tier"
        )
    
    def get_rule_performance_report(self) -> Dict[str, Any]:
        """Get detailed rule performance metrics"""
        rule_performance = {}
        
        for rule_name, metrics in self.rule_metrics.items():
            total_evaluations = max(metrics['evaluations'], 1)
            
            rule_performance[rule_name] = {
                'total_evaluations': metrics['evaluations'],
                'total_matches': metrics['matches'],
                'total_routes': metrics['routes'],
                'total_errors': metrics['errors'],
                'match_rate': metrics['matches'] / total_evaluations,
                'error_rate': metrics['errors'] / total_evaluations,
                'route_success_rate': metrics['routes'] / max(metrics['matches'], 1)
            }
        
        return {
            'adapter_type': 'selective',
            'total_rules': len(self.routing_rules),
            'rule_performance': rule_performance
        }
```

## Coordinator Integration with Event Communication

```python
class EventCommunicationFactory:
    """Factory that creates logging-aware adapters"""
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        # Create factory logger
        self.logger = ContainerLogger(
            coordinator_id,
            "communication_factory",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        # Registry of adapter types with logging integration
        self.adapters = {
            'pipeline': LoggingAwarePipelineCommunicationAdapter,
            'hierarchical': LoggingAwareHierarchicalCommunicationAdapter,
            'broadcast': LoggingAwareBroadcastCommunicationAdapter,
            'selective': LoggingAwareSelectiveCommunicationAdapter
        }
        
        # Track created adapters for lifecycle management
        self.active_adapters: List[Any] = []
    
    def create_communication_layer(self, config: Dict, containers: Dict[str, Container]) -> CommunicationLayer:
        """Create communication layer with full logging integration"""
        
        self.logger.log_info(
            "Creating communication layer",
            adapter_count=len(config.get('adapters', [])),
            container_count=len(containers),
            lifecycle_operation="communication_setup"
        )
        
        communication_layer = CommunicationLayer(self.coordinator_id, self.log_manager)
        
        for adapter_config in config.get('adapters', []):
            adapter_type = adapter_config['type']
            adapter_name = adapter_config.get('name', f"{adapter_type}_adapter")
            
            try:
                # Create logging-aware adapter
                adapter_class = self.adapters[adapter_type]
                adapter = adapter_class(self.coordinator_id, self.log_manager)
                
                # Configure adapter with logging
                self.configure_adapter_with_logging(adapter, adapter_config, containers)
                
                # Add to communication layer
                communication_layer.add_adapter(adapter, adapter_name)
                self.active_adapters.append(adapter)
                
                self.logger.log_info(
                    "Adapter created and configured",
                    adapter_type=adapter_type,
                    adapter_name=adapter_name,
                    lifecycle_operation="adapter_creation"
                )
                
            except Exception as e:
                self.logger.log_error(
                    "Failed to create adapter",
                    adapter_type=adapter_type,
                    adapter_name=adapter_name,
                    error=str(e),
                    lifecycle_operation="adapter_creation_error"
                )
                raise
        
        self.logger.log_info(
            "Communication layer creation completed",
            total_adapters=len(self.active_adapters),
            lifecycle_operation="communication_setup_complete"
        )
        
        return communication_layer
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication system metrics"""
        adapter_metrics = []
        
        for adapter in self.active_adapters:
            if hasattr(adapter, 'get_adapter_metrics'):
                metrics = adapter.get_adapter_metrics()
                adapter_metrics.append(metrics)
            elif hasattr(adapter, 'get_rule_performance_report'):
                metrics = adapter.get_rule_performance_report()
                adapter_metrics.append(metrics)
        
        return {
            'total_active_adapters': len(self.active_adapters),
            'adapter_metrics': adapter_metrics,
            'system_health': self._assess_communication_health(adapter_metrics)
        }
    
    def _assess_communication_health(self, adapter_metrics: List[Dict]) -> Dict[str, Any]:
        """Assess overall communication system health"""
        total_errors = sum(m.get('routing_errors', 0) for m in adapter_metrics)
        total_events = sum(m.get('events_routed', 0) for m in adapter_metrics)
        
        error_rate = total_errors / max(total_events, 1)
        
        return {
            'status': 'healthy' if error_rate < 0.01 else 'warning' if error_rate < 0.05 else 'critical',
            'total_events_routed': total_events,
            'total_errors': total_errors,
            'overall_error_rate': round(error_rate, 4),
            'average_latency_ms': np.mean([
                m.get('average_latency_ms', 0) for m in adapter_metrics if 'average_latency_ms' in m
            ]) if adapter_metrics else 0
        }


class CommunicationLayer:
    """Communication layer with integrated logging and lifecycle management"""
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        self.logger = ContainerLogger(
            coordinator_id,
            "communication_layer",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        self.adapters: Dict[str, Any] = {}
        self.adapter_health: Dict[str, Dict[str, Any]] = {}
        
    def add_adapter(self, adapter: Any, adapter_name: str):
        """Add adapter with health monitoring"""
        self.adapters[adapter_name] = adapter
        self.adapter_health[adapter_name] = {
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'last_health_check': None
        }
        
        self.logger.log_info(
            "Adapter added to communication layer",
            adapter_name=adapter_name,
            adapter_type=type(adapter).__name__,
            total_adapters=len(self.adapters),
            lifecycle_operation="adapter_registration"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all adapters"""
        health_report = {
            'communication_layer_status': 'healthy',
            'adapter_health': {},
            'total_adapters': len(self.adapters),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for adapter_name, adapter in self.adapters.items():
            try:
                # Get adapter metrics if available
                if hasattr(adapter, 'get_adapter_metrics'):
                    metrics = adapter.get_adapter_metrics()
                    error_rate = metrics.get('error_rate', 0)
                    
                    status = 'healthy' if error_rate < 0.01 else 'warning' if error_rate < 0.05 else 'critical'
                    
                    self.adapter_health[adapter_name].update({
                        'status': status,
                        'last_health_check': datetime.utcnow().isoformat(),
                        'metrics': metrics
                    })
                    
                    health_report['adapter_health'][adapter_name] = self.adapter_health[adapter_name]
                    
                    # Log health status
                    self.logger.log_debug(
                        "Adapter health check completed",
                        adapter_name=adapter_name,
                        status=status,
                        error_rate=error_rate,
                        lifecycle_operation="health_check"
                    )
                    
            except Exception as e:
                self.adapter_health[adapter_name]['status'] = 'error'
                health_report['adapter_health'][adapter_name] = {
                    'status': 'error',
                    'error': str(e),
                    'last_health_check': datetime.utcnow().isoformat()
                }
                
                self.logger.log_error(
                    "Adapter health check failed",
                    adapter_name=adapter_name,
                    error=str(e),
                    lifecycle_operation="health_check_error"
                )
        
        # Determine overall health
        adapter_statuses = [info['status'] for info in health_report['adapter_health'].values()]
        if 'critical' in adapter_statuses:
            health_report['communication_layer_status'] = 'critical'
        elif 'warning' in adapter_statuses or 'error' in adapter_statuses:
            health_report['communication_layer_status'] = 'warning'
        
        self.logger.log_info(
            "Communication layer health check completed",
            overall_status=health_report['communication_layer_status'],
            healthy_adapters=len([s for s in adapter_statuses if s == 'healthy']),
            total_adapters=len(self.adapters),
            lifecycle_operation="health_check_complete"
        )
        
        return health_report
    
    async def cleanup(self):
        """Cleanup all adapters and close logging"""
        self.logger.log_info(
            "Starting communication layer cleanup",
            adapter_count=len(self.adapters),
            lifecycle_operation="communication_cleanup"
        )
        
        for adapter_name, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'cleanup'):
                    await adapter.cleanup()
                
                self.logger.log_debug(
                    "Adapter cleaned up",
                    adapter_name=adapter_name,
                    lifecycle_operation="adapter_cleanup"
                )
                
            except Exception as e:
                self.logger.log_error(
                    "Error cleaning up adapter",
                    adapter_name=adapter_name,
                    error=str(e),
                    lifecycle_operation="adapter_cleanup_error"
                )
        
        self.logger.log_info(
            "Communication layer cleanup completed",
            lifecycle_operation="communication_cleanup_complete"
        )
        
        # Close our own logger
        self.logger.close()
```

## Integration with WorkflowCoordinator

```python
class WorkflowCoordinator:
    def __init__(self, config: Dict[str, Any]):
        # Initialize log manager first (as you already have)
        self.log_manager = LogManager(...)
        self.logger = self.log_manager.system_logger
        
        # Create communication factory with logging integration
        self.communication_factory = EventCommunicationFactory(
            self.coordinator_id, 
            self.log_manager
        )
        
        # Communication layer will be created when needed
        self.communication_layer = None
        
    async def setup_communication(self, communication_config: Dict):
        """Setup event communication with full logging integration"""
        
        self.logger.log_info(
            "Setting up event communication system",
            config_type=communication_config.get('pattern', 'default'),
            lifecycle_operation="communication_initialization"
        )
        
        try:
            # Create communication layer with logging
            self.communication_layer = self.communication_factory.create_communication_layer(
                communication_config, 
                self.containers
            )
            
            self.logger.log_info(
                "Event communication system initialized",
                adapter_count=len(self.communication_layer.adapters),
                lifecycle_operation="communication_ready"
            )
            
        except Exception as e:
            self.logger.log_error(
                "Failed to initialize event communication",
                error=str(e),
                lifecycle_operation="communication_initialization_error"
            )
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including communication and logging"""
        
        # Get base status (as you already have)
        status = await super().get_system_status()
        
        # Add communication metrics
        if self.communication_layer:
            try:
                communication_health = await self.communication_layer.health_check()
                communication_metrics = self.communication_factory.get_communication_metrics()
                
                status['communication'] = {
                    'health': communication_health,
                    'metrics': communication_metrics,
                    'status': communication_health['communication_layer_status']
                }
                
            except Exception as e:
                self.logger.log_error(
                    "Error getting communication status",
                    error=str(e),
                    lifecycle_operation="status_error"
                )
                
                status['communication'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    async def shutdown(self):
        """Graceful shutdown with communication cleanup"""
        
        self.logger.log_info(
            "Starting coordinator shutdown with communication cleanup",
            lifecycle_operation="coordinator_shutdown"
        )
        
        # Cleanup communication layer first
        if self.communication_layer:
            await self.communication_layer.cleanup()
        
        # Then proceed with normal shutdown (as you already have)
        await super().shutdown()
```

## Benefits of This Integration

### 1. **Complete Event Traceability**
- Every event routing decision is logged with correlation IDs
- End-to-end signal tracking across adapters and containers
- Clear audit trail for debugging communication issues

### 2. **Performance Monitoring**
- Adapter-level performance metrics (latency, throughput, error rates)
- Rule-level performance for selective routing
- Automatic health checks and alerting

### 3. **Operational Oversight**
- Communication layer status integrated with system dashboard
- Automated cleanup and lifecycle management
- Error handling and graceful degradation

### 4. **Debugging Support**
- Detailed routing logs help debug signal flow issues
- Rule evaluation logs show why events were/weren't routed
- Correlation tracking across adapter boundaries

### 5. **Production Readiness**
- Same logging lifecycle management as containers
- Automated archiving and retention policies
- Resource monitoring and alerting integration

This approach ensures that event communication adapters work seamlessly with your sophisticated logging system, maintaining the same level of observability and operational maturity that you've built for container isolation.