"""
Pipeline adapter implementation using protocol-based design.

This module implements the pipeline communication pattern without
inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

from ..events.types import Event, EventType
from .protocols import Container, CommunicationAdapter
from .helpers import (
    handle_event_with_metrics, 
    subscribe_to_container_events,
    validate_adapter_config,
    create_forward_handler
)


class PipelineAdapter:
    """Pipeline adapter - no inheritance needed!
    
    Routes events sequentially through a series of containers.
    Each container's output becomes the next container's input.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize pipeline adapter.
        
        Args:
            name: Unique adapter name
            config: Configuration with 'containers' list
        """
        self.name = name
        self.config = config
        self.containers = config.get('containers', [])
        self.connections: List[Tuple[Container, Container]] = []
        self.allow_skip = config.get('allow_skip', False)
        
        # Validate configuration
        validate_adapter_config(config, ['containers'], 'Pipeline')
        
        # Logger will be attached by factory/helper
        self.logger = logging.getLogger(f"adapter.{name}")
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure pipeline connections.
        
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
            self.connections.append((source, target))
            
        self.logger.info(
            f"Pipeline adapter '{self.name}' configured with {len(self.connections)} connections"
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
            f"Pipeline adapter '{self.name}' configured with {len(self.connections)} connections via setup_pipeline"
        )
        
    def start(self) -> None:
        """Start pipeline operation by setting up subscriptions."""
        # Set up forward connections
        for source, target in self.connections:
            # Create forwarding handler
            handler = create_forward_handler(self, target)
            
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
            
        self.logger.info(f"Pipeline adapter '{self.name}' started with forward and reverse routing")
        
    def stop(self) -> None:
        """Stop pipeline operation."""
        # In a real implementation, we'd unsubscribe here
        self.logger.info(f"Pipeline adapter '{self.name}' stopped")
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics.
        
        This method provides compatibility with the CommunicationAdapter protocol.
        It delegates to route_event for actual routing logic.
        
        Args:
            event: Event to handle
            source: Source container
        """
        handle_event_with_metrics(self, event, source)
        
    def route_event(self, event: Event, source: Container) -> None:
        """Route event to next container in pipeline.
        
        Args:
            event: Event to route
            source: Source container
        """
        # Find the source in our connections
        for i, (conn_source, conn_target) in enumerate(self.connections):
            if conn_source.name == source.name:
                # Check if we should skip this stage
                if self.allow_skip and event.metadata.get('skip_stage'):
                    self.logger.debug(
                        f"Skipping stage {conn_target.name} for event {event.event_type}"
                    )
                    # Find next target
                    if i + 1 < len(self.connections):
                        _, next_target = self.connections[i + 1]
                        self.route_event(event, conn_target)
                else:
                    # Normal forwarding
                    self.logger.debug(
                        f"Forwarding {event.event_type} from {source.name} to {conn_target.name}"
                    )
                    conn_target.receive_event(event)
                break
    
    def _setup_reverse_routing(self) -> None:
        """Set up reverse routing for specific event types like FILL and broadcast routing for SYSTEM events."""
        # Find containers by role
        containers_by_role = {}
        all_containers = []
        
        # Collect all containers from connections
        for source, target in self.connections:
            if hasattr(source, 'metadata') and hasattr(source.metadata, 'role'):
                role = source.metadata.role.value
                containers_by_role[role] = source
                if source not in all_containers:
                    all_containers.append(source)
            if hasattr(target, 'metadata') and hasattr(target.metadata, 'role'):
                role = target.metadata.role.value
                containers_by_role[role] = target
                if target not in all_containers:
                    all_containers.append(target)
        
        # Also check containers passed to setup_pipeline that might not be in connections
        if hasattr(self, '_pipeline_containers'):
            for container in self._pipeline_containers:
                if hasattr(container, 'metadata') and hasattr(container.metadata, 'role'):
                    role = container.metadata.role.value
                    if role not in containers_by_role:
                        containers_by_role[role] = container
                        if container not in all_containers:
                            all_containers.append(container)
        
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


def create_conditional_pipeline(name: str, config: Dict[str, Any]):
    """Factory function for conditional pipeline variant.
    
    This creates a pipeline that can skip stages based on conditions.
    
    Args:
        name: Adapter name
        config: Configuration with conditions
        
    Returns:
        Configured pipeline adapter
    """
    # Extract conditions from config
    conditions = config.get('conditions', [])
    
    # Create base pipeline
    pipeline = PipelineAdapter(name, config)
    
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
        name: Adapter name  
        config: Configuration with parallel pipelines
        
    Returns:
        Composite adapter managing parallel pipelines
    """
    pipelines = []
    pipeline_configs = config.get('pipelines', [])
    
    for i, pipeline_spec in enumerate(pipeline_configs):
        sub_config = {
            'containers': pipeline_spec,
            'allow_skip': config.get('allow_skip', False)
        }
        sub_pipeline = PipelineAdapter(f"{name}_parallel_{i}", sub_config)
        pipelines.append(sub_pipeline)
    
    # Create composite adapter
    class ParallelPipelineAdapter:
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
    
    return ParallelPipelineAdapter()