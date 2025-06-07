"""
Pipeline route implementation using protocol-based design.

This module implements the pipeline communication pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Tuple, Optional, Type
import logging

from ..types.events import Event, EventType
from ..events.semantic import SemanticEvent, validate_semantic_event
from ..events.type_flow_analysis import TypeFlowAnalyzer, EventTypeRegistry, ContainerTypeInferencer
from .protocols import Container, CommunicationRoute
from .composition import (
    wrap_with_metrics,
    create_subscription,
    validate_config,
    create_forwarding_handler
)


class PipelineRoute:
    """Pipeline route - no inheritance needed!
    
    Routes events sequentially through a series of containers.
    Each container's output becomes the next container's input.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize pipeline route.
        
        Args:
            name: Unique route name
            config: Configuration with 'containers' list
        """
        self.name = name
        self.config = config
        self.containers = config.get('containers', [])
        self.connections: List[Tuple[Container, Container]] = []
        self.allow_skip = config.get('allow_skip', False)
        self._is_started = False
        
        # Type flow analysis components
        self.enable_type_validation = config.get('enable_type_validation', True)
        self.registry = EventTypeRegistry()
        self.type_analyzer = TypeFlowAnalyzer(self.registry)
        self.type_inferencer = ContainerTypeInferencer(self.registry)
        
        # Validate configuration
        validate_config(config, ['containers'], 'Pipeline')
        
        # Logger will be attached by factory/helper
        self.logger = logging.getLogger(f"route.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure pipeline connections with type flow validation.
        
        Args:
            containers: Map of container names to instances
        """
        # Build connection pairs
        for i in range(len(self.containers) - 1):
            source_name = self.containers[i]
            target_name = self.containers[i + 1]
            
            if source_name not in containers:
                raise ValueError(f"Container '{source_name}' not found")
            if target_name not in containers:
                raise ValueError(f"Container '{target_name}' not found")
                
            source = containers[source_name]
            target = containers[target_name]
            
            # Validate type flow compatibility
            if self.enable_type_validation:
                self._validate_connection(source, target)
            
            self.connections.append((source, target))
            
        # Perform full pipeline type flow analysis
        if self.enable_type_validation:
            self._validate_pipeline_flow(containers)
            
        self.logger.info(
            f"Pipeline route '{self.name}' configured with {len(self.connections)} connections"
        )
        
    def setup_pipeline(self, container_list: List[Container]) -> None:
        """Alternative setup method that takes a list of containers in order.
        
        This method is for compatibility with code expecting setup_pipeline.
        
        Args:
            container_list: List of containers in pipeline order
        """
        # Store all containers for reverse routing
        self._pipeline_containers = container_list
        
        # Clear existing connections
        self.connections = []
        
        # Build connections from ordered list
        for i in range(len(container_list) - 1):
            source = container_list[i]
            target = container_list[i + 1]
            self.connections.append((source, target))
            
        self.logger.info(
            f"Pipeline route '{self.name}' configured with {len(self.connections)} connections via setup_pipeline"
        )
        
    def start(self) -> None:
        """Start pipeline operation by setting up subscriptions."""
        if self._is_started:
            self.logger.warning(f"Pipeline route '{self.name}' is already started")
            return
            
        # Don't start if no connections configured
        if not self.connections:
            self.logger.info(f"Pipeline route '{self.name}' has no connections, skipping start")
            return
            
        # Set up forward connections
        for source, target in self.connections:
            # Create forwarding handler
            handler = create_forwarding_handler(target, self.logger)
            
            # Subscribe using the container's on_output_event method if available
            if hasattr(source, 'on_output_event'):
                source.on_output_event(handler)
            elif hasattr(source, 'event_bus') and hasattr(source.event_bus, 'subscribe_all'):
                source.event_bus.subscribe_all(handler)
            else:
                self.logger.warning(f"Container {source.name} doesn't support event subscription")
        
        # Set up reverse routing for FILL events
        # ExecutionContainer needs to send FILL events back to RiskContainer and PortfolioContainer
        self._setup_reverse_routing()
        
        self._is_started = True
        self.logger.info(f"Pipeline route '{self.name}' started with forward and reverse routing")
        
    def stop(self) -> None:
        """Stop pipeline operation."""
        if not self._is_started:
            self.logger.info(f"Pipeline route '{self.name}' is already stopped")
            return
            
        # In a real implementation, we'd unsubscribe here
        self._is_started = False
        self.logger.info(f"Pipeline route '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics.
        
        This method provides compatibility with the CommunicationRoute protocol.
        It delegates to route_event for actual routing logic.
        
        Args:
            event: Event to handle
            source: Source container
        """
        # Wrap the route_event method with metrics if available
        if hasattr(self, 'metrics'):
            wrapped = wrap_with_metrics(self.route_event, self)
            wrapped(event, source)
        else:
            self.route_event(event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Route event to next container in pipeline with type validation.
        
        Args:
            event: Event to route
            source: Source container
        """
        # Find the source in our connections
        for i, (conn_source, conn_target) in enumerate(self.connections):
            if conn_source.name == source.name:
                # Validate event before routing
                if self.enable_type_validation:
                    self._validate_event_routing(event, source, conn_target)
                
                # Check if we should skip this stage
                if self.allow_skip and hasattr(event, 'metadata') and event.metadata.get('skip_stage'):
                    self.logger.debug(
                        f"Skipping stage {conn_target.name} for event {getattr(event, 'event_type', type(event).__name__)}"
                    )
                    # Find next target
                    if i + 1 < len(self.connections):
                        _, next_target = self.connections[i + 1]
                        self.route_event(event, conn_target)
                else:
                    # Normal forwarding
                    self.logger.debug(
                        f"Forwarding {getattr(event, 'event_type', type(event).__name__)} from {source.name} to {conn_target.name}"
                    )
                    conn_target.receive_event(event)
                break
    
    def _setup_reverse_routing(self) -> None:
        """Set up reverse routing for specific event types like FILL and broadcast routing for SYSTEM events."""
        # Find containers by role
        containers_by_role = {}
        all_containers = []
        
        # Helper function to collect all containers including nested ones
        def collect_all_containers(container, collected=None):
            if collected is None:
                collected = []
            if container not in collected:
                collected.append(container)
                if hasattr(container, 'metadata') and hasattr(container.metadata, 'role'):
                    role = container.metadata.role.value
                    containers_by_role[role] = container
                # Recursively collect child containers
                if hasattr(container, 'child_containers'):
                    for child in container.child_containers:
                        collect_all_containers(child, collected)
            return collected
        
        # Collect all containers from connections (including nested)
        for source, target in self.connections:
            collect_all_containers(source, all_containers)
            collect_all_containers(target, all_containers)
        
        # Also check containers passed to setup_pipeline that might not be in connections
        if hasattr(self, '_pipeline_containers'):
            for container in self._pipeline_containers:
                collect_all_containers(container, all_containers)
        
        self.logger.info(f"Found containers by role: {list(containers_by_role.keys())}")
        
        # Set up FILL event routing from ExecutionContainer back to Portfolio and Risk
        execution_container = containers_by_role.get('execution')
        portfolio_container = containers_by_role.get('portfolio')
        risk_container = containers_by_role.get('risk')
        data_container = containers_by_role.get('data')
        
        if execution_container and (portfolio_container or risk_container):
            # Create handler that routes FILL events backward
            def route_fill_backward(event: Event):
                if event.event_type == EventType.FILL:
                    self.logger.debug(f"Routing FILL event backward from execution")
                    # Send to portfolio first (if exists)
                    if portfolio_container:
                        portfolio_container.receive_event(event)
                    # Then to risk container
                    if risk_container:
                        risk_container.receive_event(event)
            
            # Subscribe to FILL events from execution container
            if hasattr(execution_container, 'event_bus'):
                execution_container.event_bus.subscribe(EventType.FILL, route_fill_backward)
                self.logger.info(f"Set up reverse routing for FILL events from ExecutionContainer")
        
        # Set up PORTFOLIO event routing from PortfolioContainer to RiskContainer
        if portfolio_container and risk_container:
            def route_portfolio_to_risk(event: Event):
                if event.event_type == EventType.PORTFOLIO:
                    self.logger.debug(f"Routing PORTFOLIO event from Portfolio to Risk")
                    risk_container.receive_event(event)
            
            # Subscribe to PORTFOLIO events from portfolio container
            if hasattr(portfolio_container, 'event_bus'):
                portfolio_container.event_bus.subscribe(EventType.PORTFOLIO, route_portfolio_to_risk)
                self.logger.info(f"Set up PORTFOLIO event routing from PortfolioContainer to RiskContainer")
        else:
            if not portfolio_container:
                self.logger.warning("No PortfolioContainer found for PORTFOLIO event routing")
            if not risk_container:
                self.logger.warning("No RiskContainer found for PORTFOLIO event routing")
        
        # Set up SYSTEM event broadcasting from DataContainer to all containers
        if data_container:
            def broadcast_system_event(event: Event):
                if event.event_type == EventType.SYSTEM:
                    message = event.payload.get('message', '')
                    self.logger.info(f"Broadcasting SYSTEM event '{message}' to all containers")
                    # Send to all containers that can handle system events
                    for container in all_containers:
                        if container != data_container:  # Don't send back to source
                            container.receive_event(event)
            
            # Subscribe to SYSTEM events from data container
            if hasattr(data_container, 'event_bus'):
                data_container.event_bus.subscribe(EventType.SYSTEM, broadcast_system_event)
                self.logger.info(f"Set up SYSTEM event broadcasting from DataContainer")
    
    def _validate_connection(self, source: Container, target: Container) -> None:
        """Validate that target can handle source's event types.
        
        Args:
            source: Source container
            target: Target container
            
        Raises:
            TypeError: If type flow is invalid
        """
        try:
            # Get expected event types for each container
            source_outputs = self.type_inferencer.get_expected_outputs(source)
            target_inputs = self.type_inferencer.get_expected_inputs(target)
            
            if source_outputs and target_inputs:
                # Check if any source output can be handled by target
                compatible_types = source_outputs & target_inputs
                if not compatible_types:
                    source_type = self.type_inferencer.infer_container_type(source)
                    target_type = self.type_inferencer.infer_container_type(target)
                    
                    self.logger.warning(
                        f"Type flow warning: {source.name} ({source_type}) outputs "
                        f"{[t.__name__ for t in source_outputs]} but {target.name} ({target_type}) "
                        f"expects {[t.__name__ for t in target_inputs]}"
                    )
                    
                    # Don't raise exception for warning - let system try to work
                    # In strict mode, we could raise TypeError here
                    if self.config.get('strict_type_validation', False):
                        raise TypeError(
                            f"Type flow error: {target.name} cannot handle any events from {source.name}"
                        )
                else:
                    self.logger.debug(
                        f"Type flow OK: {source.name} → {target.name} "
                        f"(compatible: {[t.__name__ for t in compatible_types]})"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error validating connection {source.name} → {target.name}: {e}")
            if self.config.get('strict_type_validation', False):
                raise
    
    def _validate_pipeline_flow(self, containers: Dict[str, Container]) -> None:
        """Validate the complete pipeline type flow.
        
        Args:
            containers: All available containers
        """
        try:
            # Build flow map for this pipeline
            pipeline_containers = {name: containers[name] for name in self.containers if name in containers}
            flow_map = self.type_analyzer.analyze_flow(pipeline_containers, [self])
            
            # Validate for appropriate execution mode
            execution_mode = self.config.get('execution_mode', 'full_backtest')
            validation_result = self.type_analyzer.validate_mode(flow_map, execution_mode)
            
            if not validation_result.valid:
                error_msg = f"Pipeline type flow validation failed: {'; '.join(validation_result.errors)}"
                self.logger.error(error_msg)
                
                if self.config.get('strict_type_validation', False):
                    raise TypeError(error_msg)
            else:
                self.logger.info(f"Pipeline type flow validation passed for mode '{execution_mode}'")
                
            # Log warnings
            for warning in validation_result.warnings:
                self.logger.warning(f"Type flow warning: {warning}")
                
        except Exception as e:
            self.logger.error(f"Error validating pipeline flow: {e}")
            if self.config.get('strict_type_validation', False):
                raise
    
    def _validate_event_routing(self, event: Any, source: Container, target: Container) -> None:
        """Validate that a specific event can be routed from source to target.
        
        Args:
            event: Event to validate
            source: Source container
            target: Target container
            
        Raises:
            TypeError: If event cannot be routed
        """
        try:
            # Validate semantic event if applicable
            if isinstance(event, SemanticEvent):
                if not validate_semantic_event(event):
                    self.logger.warning(f"Invalid semantic event: {event}")
                    
                # Check if target can handle this semantic event type
                target_inputs = self.type_inferencer.get_expected_inputs(target)
                event_type = type(event)
                
                if target_inputs and event_type not in target_inputs:
                    self.logger.warning(
                        f"Event type mismatch: {target.name} expects "
                        f"{[t.__name__ for t in target_inputs]} but got {event_type.__name__}"
                    )
                    
                    if self.config.get('strict_type_validation', False):
                        raise TypeError(
                            f"Event type error: {target.name} cannot handle {event_type.__name__}"
                        )
            
            # Additional validation for traditional Event objects
            elif hasattr(event, 'event_type'):
                # Get EventType from registry if possible
                event_type = getattr(event, 'event_type', None)
                if event_type:
                    # This would require container interface to declare supported event types
                    # For now, just log the event type being routed
                    self.logger.debug(
                        f"Routing {event_type} event from {source.name} to {target.name}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error validating event routing: {e}")
            if self.config.get('strict_type_validation', False):
                raise


def create_conditional_pipeline(name: str, config: Dict[str, Any]):
    """Factory function for conditional pipeline variant.
    
    This creates a pipeline that can skip stages based on conditions.
    
    Args:
        name: Route name
        config: Configuration with conditions
        
    Returns:
        Configured pipeline route
    """
    # Extract conditions from config
    conditions = config.get('conditions', [])
    
    # Create base pipeline
    pipeline = PipelineRoute(name, config)
    
    # Extend with conditional logic
    original_route = pipeline.route_event
    
    def conditional_route(event: Event, source: Container) -> None:
        """Route with condition checking."""
        # Check conditions
        for condition in conditions:
            stage = condition.get('stage')
            skip_expr = condition.get('skip_if')
            
            # Simple condition evaluation (in production, use safe eval)
            if stage in pipeline.containers:
                try:
                    # This is simplified - real implementation needs safe evaluation
                    if eval(skip_expr, {"event": event}):
                        event.metadata['skip_stage'] = stage
                except Exception as e:
                    if hasattr(pipeline, 'logger'):
                        pipeline.logger.warning(f"Error evaluating condition: {e}")
        
        # Call original routing
        original_route(event, source)
    
    pipeline.route_event = conditional_route
    return pipeline


def create_parallel_pipeline(name: str, config: Dict[str, Any]):
    """Factory function for parallel pipeline variant.
    
    This creates multiple pipeline instances that process in parallel.
    
    Args:
        name: Route name  
        config: Configuration with parallel pipelines
        
    Returns:
        Composite route managing parallel pipelines
    """
    pipelines = []
    pipeline_configs = config.get('pipelines', [])
    
    for i, pipeline_spec in enumerate(pipeline_configs):
        sub_config = {
            'containers': pipeline_spec,
            'allow_skip': config.get('allow_skip', False)
        }
        sub_pipeline = PipelineRoute(f"{name}_parallel_{i}", sub_config)
        pipelines.append(sub_pipeline)
    
    # Create composite route
    class ParallelPipelineRoute:
        def __init__(self):
            self.name = name
            self.config = config
            self.pipelines = pipelines
            
        def setup(self, containers: Dict[str, Container]) -> None:
            for pipeline in self.pipelines:
                pipeline.setup(containers)
                
        def start(self) -> None:
            for pipeline in self.pipelines:
                pipeline.start()
                
        def stop(self) -> None:
            for pipeline in self.pipelines:
                pipeline.stop()
                
        def handle_event(self, event: Event, source: Container) -> None:
            # Route to all parallel pipelines
            for pipeline in self.pipelines:
                pipeline.handle_event(event, source)
    
    return ParallelPipelineRoute()