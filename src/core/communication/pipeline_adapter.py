"""Pipeline Communication Adapter for ADMF-PC.

This module implements a linear pipeline communication adapter that processes
events through a sequence of containers with optional transformation support.

Key features:
- Linear event flow through container stages
- Event transformation between stages
- Correlation ID tracking throughout pipeline
- Stage-by-stage latency tracking
- Comprehensive error handling and recovery
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
import time
import uuid
from datetime import datetime

from .base_adapter import CommunicationAdapter, AdapterConfig
from ..logging.container_logger import ContainerLogger
from ..events.types import Event, EventType


class EventTransformer:
    """Handles event transformation between pipeline stages."""
    
    def __init__(self, logger: Optional[ContainerLogger] = None):
        """Initialize the event transformer.
        
        Args:
            logger: Logger for transformer operations
        """
        self.logger = logger
        self.transformation_rules: Dict[tuple, Callable] = {
            # Default transformation rules
            (EventType.BAR, EventType.INDICATOR): self._transform_bar_to_indicator,
            (EventType.SIGNAL, EventType.ORDER): self._transform_signal_to_order,
            (EventType.ORDER, EventType.FILL): self._transform_order_to_fill,
            (EventType.INDICATOR, EventType.SIGNAL): self._transform_indicator_to_signal,
        }
        
    def add_rule(self, from_type: EventType, to_type: EventType, transformer: Callable):
        """Add a custom transformation rule.
        
        Args:
            from_type: Source event type
            to_type: Target event type
            transformer: Transformation function
        """
        self.transformation_rules[(from_type, to_type)] = transformer
        if self.logger:
            self.logger.debug(
                "Added transformation rule",
                from_type=from_type.name,
                to_type=to_type.name
            )
    
    def transform(self, event: Event, source_container: Any, target_container: Any) -> Event:
        """Transform event based on source and target container requirements.
        
        Args:
            event: Event to transform
            source_container: Source container
            target_container: Target container
            
        Returns:
            Transformed event (or original if no transformation needed)
        """
        # Get expected input type from target container
        target_input_type = getattr(target_container, 'expected_input_type', None)
        if not target_input_type:
            return event  # Target accepts any event type
        
        # Check if transformation is needed
        if isinstance(event.event_type, EventType) and isinstance(target_input_type, EventType):
            transformation_key = (event.event_type, target_input_type)
            if transformation_key in self.transformation_rules:
                try:
                    transformed = self.transformation_rules[transformation_key](event)
                    if self.logger:
                        self.logger.debug(
                            "Event transformed",
                            from_type=event.event_type.name,
                            to_type=target_input_type.name,
                            event_id=event.metadata.get('event_id', 'unknown')
                        )
                    return transformed
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            "Event transformation failed",
                            error=str(e),
                            from_type=event.event_type.name,
                            to_type=target_input_type.name
                        )
                    # Return original event on transformation failure
                    return event
        
        return event
    
    def _transform_bar_to_indicator(self, event: Event) -> Event:
        """Transform BAR event to INDICATOR event."""
        return Event(
            event_type=EventType.INDICATOR,
            payload={
                'symbol': event.payload.get('symbol'),
                'price': event.payload.get('price'),
                'volume': event.payload.get('volume'),
                'indicators': {}  # Placeholder for calculated indicators
            },
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata={**event.metadata, 'transformed_from': 'BAR'}
        )
    
    def _transform_signal_to_order(self, event: Event) -> Event:
        """Transform SIGNAL event to ORDER event."""
        signal_data = event.payload
        return Event(
            event_type=EventType.ORDER,
            payload={
                'symbol': signal_data.get('symbol'),
                'side': signal_data.get('side', 'BUY'),
                'quantity': signal_data.get('quantity', 0),
                'order_type': 'MARKET',  # Default to market order
                'signal_id': event.metadata.get('event_id'),
                'signal_confidence': signal_data.get('confidence', 0.0)
            },
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata={**event.metadata, 'transformed_from': 'SIGNAL'}
        )
    
    def _transform_order_to_fill(self, event: Event) -> Event:
        """Transform ORDER event to FILL event."""
        order_data = event.payload
        return Event(
            event_type=EventType.FILL,
            payload={
                'symbol': order_data.get('symbol'),
                'side': order_data.get('side'),
                'quantity': order_data.get('quantity'),
                'fill_price': order_data.get('price', 0.0),  # Simulated fill
                'order_id': event.metadata.get('event_id'),
                'timestamp': datetime.now()
            },
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata={**event.metadata, 'transformed_from': 'ORDER'}
        )
    
    def _transform_indicator_to_signal(self, event: Event) -> Event:
        """Transform INDICATOR event to SIGNAL event."""
        indicator_data = event.payload
        return Event(
            event_type=EventType.SIGNAL,
            payload={
                'symbol': indicator_data.get('symbol'),
                'side': 'BUY',  # Placeholder logic
                'confidence': 0.5,  # Placeholder confidence
                'indicators': indicator_data.get('indicators', {})
            },
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata={**event.metadata, 'transformed_from': 'INDICATOR'}
        )


class PipelineStage:
    """Represents a single stage in the pipeline."""
    
    def __init__(self, container: Any, stage_number: int, next_stage: Optional['PipelineStage'] = None):
        """Initialize a pipeline stage.
        
        Args:
            container: Container for this stage
            stage_number: Stage position in pipeline (0-based)
            next_stage: Next stage in the pipeline
        """
        self.container = container
        self.stage_number = stage_number
        self.next_stage = next_stage
        self.container_id = getattr(container, 'container_id', f'stage_{stage_number}')
        
        # Stage metrics
        self.events_processed = 0
        self.events_failed = 0
        self.total_latency_ms = 0.0
        

class PipelineCommunicationAdapter(CommunicationAdapter):
    """Linear pipeline communication adapter for sequential event processing.
    
    This adapter wires containers in a linear sequence where events flow
    from one container to the next with optional transformation support.
    """
    
    def __init__(self, config: AdapterConfig, logger: Optional[ContainerLogger] = None):
        """Initialize the pipeline adapter.
        
        Args:
            config: Adapter configuration
            logger: Container logger
        """
        super().__init__(config, logger)
        
        self.pipeline_stages: List[PipelineStage] = []
        self.event_transformer = EventTransformer(logger)
        self._stage_handlers: Dict[str, Callable] = {}
        
        # Pipeline-specific metrics
        self.stage_metrics: Dict[int, Dict[str, Any]] = {}
        
    async def connect(self) -> bool:
        """Establish pipeline connections.
        
        Returns:
            True if pipeline setup successful
        """
        try:
            # Pipeline doesn't need external connections
            # It uses direct container references
            self.logger.info(
                "Pipeline adapter connected",
                pipeline_length=len(self.pipeline_stages)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to connect pipeline adapter",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
    
    async def disconnect(self) -> None:
        """Disconnect pipeline connections."""
        try:
            # Unwire all pipeline stages
            for stage in self.pipeline_stages:
                container = stage.container
                if hasattr(container, 'remove_output_handler') and stage.container_id in self._stage_handlers:
                    container.remove_output_handler(self._stage_handlers[stage.container_id])
            
            # Clean up reverse routing handlers
            for handler_key in list(self._stage_handlers.keys()):
                if handler_key.startswith('reverse_fill_'):
                    # Find the execution container and unsubscribe the reverse handler
                    for stage in self.pipeline_stages:
                        container = stage.container
                        if (hasattr(container, 'metadata') and 
                            hasattr(container.metadata, 'role') and 
                            container.metadata.role.value == 'execution' and
                            hasattr(container, 'event_bus')):
                            
                            handler = self._stage_handlers[handler_key]
                            container.event_bus.unsubscribe(EventType.FILL, handler)
                            self.logger.info("Cleaned up reverse FILL routing handler")
                            break
            
            self._stage_handlers.clear()
            self.pipeline_stages.clear()
            
            self.logger.info("Pipeline adapter disconnected")
            
        except Exception as e:
            self.logger.error(
                "Error disconnecting pipeline adapter",
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def send_raw(self, data: bytes, correlation_id: Optional[str] = None) -> bool:
        """Pipeline adapter doesn't send raw bytes - it routes events directly.
        
        Args:
            data: Raw bytes (not used)
            correlation_id: Correlation ID
            
        Returns:
            False (not supported for pipeline)
        """
        self.logger.warning(
            "send_raw not supported for pipeline adapter",
            correlation_id=correlation_id
        )
        return False
    
    async def receive_raw(self) -> Optional[bytes]:
        """Pipeline adapter doesn't receive raw bytes - it routes events directly.
        
        Returns:
            None (not supported for pipeline)
        """
        return None
    
    def setup_pipeline(self, containers: List[Any]) -> None:
        """Setup the pipeline with a list of containers.
        
        Args:
            containers: List of containers in pipeline order
        """
        if not containers:
            raise ValueError("Pipeline requires at least one container")
        
        self.logger.info(
            "🔧 Starting pipeline setup",
            container_count=len(containers),
            container_ids=[getattr(c, 'container_id', f'container_{i}') for i, c in enumerate(containers)]
        )
        
        # Create pipeline stages
        self.pipeline_stages = []
        for i, container in enumerate(containers):
            stage = PipelineStage(container, i)
            self.pipeline_stages.append(stage)
            
            # Initialize stage metrics
            self.stage_metrics[i] = {
                'events_processed': 0,
                'events_failed': 0,
                'total_latency_ms': 0.0,
                'average_latency_ms': 0.0,
                'container_id': stage.container_id
            }
        
        # Link stages
        for i in range(len(self.pipeline_stages) - 1):
            self.pipeline_stages[i].next_stage = self.pipeline_stages[i + 1]
        
        # Wire up forward event handlers (normal pipeline flow)
        for i in range(len(self.pipeline_stages) - 1):
            source_stage = self.pipeline_stages[i]
            target_stage = self.pipeline_stages[i + 1]
            self._wire_pipeline_stage(source_stage.container, target_stage.container, i)
        
        # Wire up reverse event handlers for FILL events (Execution -> Portfolio)
        self.logger.info("🔄 Setting up reverse FILL routing...")
        self._setup_reverse_fill_routing()
        
        # Wire up PORTFOLIO events (Portfolio -> Risk)
        self.logger.info("🔄 Setting up PORTFOLIO event routing...")
        self._setup_portfolio_routing()
        
        # Wire up SYSTEM event broadcasting from DataContainer
        self.logger.info("🔄 Setting up SYSTEM event broadcasting...")
        self._setup_system_event_broadcasting()
        
        self.logger.info(
            "Pipeline setup complete",
            total_stages=len(self.pipeline_stages),
            wired_stages=len(self.pipeline_stages) - 1,
            reverse_routes_configured=True,
            system_event_broadcasting=True
        )
    
    def _wire_pipeline_stage(self, source_container, target_container, stage_index: int) -> None:
        """Wire pipeline stage to route events from source to target container.
        
        Args:
            source_container: Container that produces events
            target_container: Container that receives events
            stage_index: Index of this stage in the pipeline
        """
        def stage_handler(event: Event) -> None:
            """Handle events from source container and forward to target."""
            # Special handling for SYSTEM events - broadcast to all containers
            if event.event_type == EventType.SYSTEM:
                self._broadcast_system_event(event)
                return
            
            # Transform event if needed
            transformed_event = self.event_transformer.transform(event, source_container, target_container)
            if transformed_event and hasattr(target_container, 'receive_event'):
                target_container.receive_event(transformed_event)
                
                # Update metrics
                self.metrics.events_sent += 1
                self.logger.debug(
                    f"Event forwarded: {event.event_type} -> {target_container.metadata.name}"
                )
        
        # Register the handler with the source container's output events
        if hasattr(source_container, 'on_output_event'):
            source_container.on_output_event(stage_handler)
            self.logger.info(
                f"Wired pipeline stage {stage_index}: {source_container.metadata.name} -> {target_container.metadata.name}"
            )
        else:
            self.logger.warning(
                f"Source container {source_container.metadata.name} doesn't support on_output_event"
            )
        
        # Store handler reference for cleanup
        self._stage_handlers[source_container.metadata.container_id] = stage_handler
    
    def _setup_reverse_fill_routing(self) -> None:
        """Setup reverse routing for FILL events from ExecutionContainer to PortfolioContainer."""
        # Find ExecutionContainer and PortfolioContainer in pipeline
        execution_container = None
        portfolio_container = None
        
        for stage in self.pipeline_stages:
            container = stage.container
            if hasattr(container, 'metadata'):
                role = getattr(container.metadata, 'role', None)
                if role and hasattr(role, 'value'):
                    role_value = role.value
                elif role:
                    role_value = str(role)
                else:
                    role_value = 'unknown'
                
                self.logger.info(
                    f"Checking container for reverse routing: {container.metadata.name} with role: {role_value}"
                )
                
                if role_value == 'execution':
                    execution_container = container
                    self.logger.info(f"Found ExecutionContainer: {container.metadata.name}")
                elif role_value == 'portfolio':
                    portfolio_container = container
                    self.logger.info(f"Found PortfolioContainer: {container.metadata.name}")
        
        if execution_container and portfolio_container:
            self.logger.info(
                "Setting up reverse FILL routing",
                from_container=execution_container.metadata.name,
                to_container=portfolio_container.metadata.name
            )
            
            def fill_reverse_handler(event: Event) -> None:
                """Handle FILL events and route them back to PortfolioContainer."""
                if event.event_type == EventType.FILL:
                    self.logger.info(
                        f"🔄 Reverse routing FILL event: {execution_container.metadata.name} -> {portfolio_container.metadata.name}"
                    )
                    
                    # Send FILL event directly to PortfolioContainer
                    if hasattr(portfolio_container, 'receive_event'):
                        portfolio_container.receive_event(event)
                        self.metrics.events_sent += 1
                    else:
                        self.logger.warning(
                            f"PortfolioContainer {portfolio_container.metadata.name} doesn't support receive_event"
                        )
            
            # Subscribe ExecutionContainer's FILL events to the reverse handler
            if hasattr(execution_container, 'event_bus'):
                execution_container.event_bus.subscribe(EventType.FILL, fill_reverse_handler)
                self.logger.info("✅ Reverse FILL routing configured successfully")
                
                # Store handler for cleanup
                reverse_handler_key = f"reverse_fill_{execution_container.metadata.container_id}"
                self._stage_handlers[reverse_handler_key] = fill_reverse_handler
            else:
                self.logger.warning("ExecutionContainer doesn't have event_bus for reverse routing")
        else:
            containers_found = []
            if execution_container:
                containers_found.append("ExecutionContainer")
            if portfolio_container:
                containers_found.append("PortfolioContainer")
            
            self.logger.warning(
                f"Cannot setup reverse FILL routing - missing containers. Found: {containers_found}"
            )
    
    def _setup_portfolio_routing(self) -> None:
        """Setup routing for PORTFOLIO events from PortfolioContainer to RiskContainer."""
        # Find PortfolioContainer and RiskContainer in pipeline
        portfolio_container = None
        risk_container = None
        
        for stage in self.pipeline_stages:
            container = stage.container
            if hasattr(container, 'metadata'):
                role = getattr(container.metadata, 'role', None)
                if role and hasattr(role, 'value'):
                    role_value = role.value
                elif role:
                    role_value = str(role)
                else:
                    role_value = 'unknown'
                
                if role_value == 'portfolio':
                    portfolio_container = container
                    self.logger.info(f"Found PortfolioContainer: {container.metadata.name}")
                elif role_value == 'risk':
                    risk_container = container
                    self.logger.info(f"Found RiskContainer: {container.metadata.name}")
        
        if portfolio_container and risk_container:
            self.logger.info(
                "Setting up PORTFOLIO event routing",
                from_container=portfolio_container.metadata.name,
                to_container=risk_container.metadata.name
            )
            
            def portfolio_routing_handler(event: Event) -> None:
                """Route PORTFOLIO events to RiskContainer."""
                if event.event_type == EventType.PORTFOLIO:
                    self.logger.info(
                        f"📊 Routing PORTFOLIO event: {portfolio_container.metadata.name} -> {risk_container.metadata.name}"
                    )
                    
                    # Send PORTFOLIO event directly to RiskContainer
                    if hasattr(risk_container, 'receive_event'):
                        risk_container.receive_event(event)
                        self.metrics.events_sent += 1
            
            # Subscribe PortfolioContainer's PORTFOLIO events to the routing handler
            if hasattr(portfolio_container, 'event_bus'):
                portfolio_container.event_bus.subscribe(EventType.PORTFOLIO, portfolio_routing_handler)
                self.logger.info("✅ PORTFOLIO event routing configured successfully")
                
                # Store handler for cleanup
                portfolio_handler_key = f"portfolio_{portfolio_container.metadata.container_id}"
                self._stage_handlers[portfolio_handler_key] = portfolio_routing_handler
        else:
            containers_found = []
            if portfolio_container:
                containers_found.append("PortfolioContainer")
            if risk_container:
                containers_found.append("RiskContainer")
            
            self.logger.warning(
                f"Cannot setup PORTFOLIO routing - missing containers. Found: {containers_found}"
            )
    
    def _setup_system_event_broadcasting(self) -> None:
        """Setup broadcasting for SYSTEM events from DataContainer to all containers."""
        # Find DataContainer in pipeline
        data_container = None
        
        for stage in self.pipeline_stages:
            container = stage.container
            if hasattr(container, 'metadata'):
                role = getattr(container.metadata, 'role', None)
                if role and hasattr(role, 'value'):
                    role_value = role.value
                elif role:
                    role_value = str(role)
                else:
                    role_value = 'unknown'
                
                if role_value == 'data':
                    data_container = container
                    self.logger.info(f"Found DataContainer: {container.metadata.name}")
                    break
        
        if data_container:
            self.logger.info("✅ SYSTEM event broadcasting configured (handled in stage handler)")
        else:
            self.logger.warning("Cannot setup SYSTEM event broadcasting - DataContainer not found")
    
    def _broadcast_system_event(self, event: Event) -> None:
        """Broadcast SYSTEM events to all containers in the pipeline.
        
        Args:
            event: SYSTEM event to broadcast
        """
        message = event.payload.get('message', '')
        self.logger.info(
            f"📢 Broadcasting SYSTEM event '{message}' to all {len(self.pipeline_stages)} containers"
        )
        
        # Send to all containers
        for stage in self.pipeline_stages:
            container = stage.container
            if hasattr(container, 'receive_event'):
                try:
                    container.receive_event(event)
                    self.logger.debug(
                        f"SYSTEM event '{message}' sent to {container.metadata.name}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to send SYSTEM event to {container.metadata.name}",
                        error=str(e),
                        error_type=type(e).__name__
                    )
        
        # Update metrics
        self.metrics.events_sent += len(self.pipeline_stages)
    
    async def _handle_pipeline_event(self, event: Event, stage: PipelineStage) -> None:
        """Handle an event flowing through the pipeline.
        
        Args:
            event: Event to process
            stage: Current pipeline stage
        """
        if not stage.next_stage:
            return  # End of pipeline
        
        start_time = time.time()
        correlation_id = event.metadata.get('correlation_id', str(uuid.uuid4()))
        
        # Ensure correlation ID is set
        if 'correlation_id' not in event.metadata:
            event.metadata['correlation_id'] = correlation_id
        
        # Add pipeline tracking metadata
        if 'pipeline_path' not in event.metadata:
            event.metadata['pipeline_path'] = []
        event.metadata['pipeline_path'].append({
            'stage': stage.stage_number,
            'container': stage.container_id,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            self.logger.debug(
                "Processing pipeline event",
                stage_number=stage.stage_number,
                source_container=stage.container_id,
                target_container=stage.next_stage.container_id,
                event_type=event.event_type.name if isinstance(event.event_type, EventType) else event.event_type,
                correlation_id=correlation_id
            )
            
            # Transform event if needed
            transformed_event = self.event_transformer.transform(
                event, 
                stage.container, 
                stage.next_stage.container
            )
            
            # Forward to next stage
            next_container = stage.next_stage.container
            if hasattr(next_container, 'receive_event'):
                await next_container.receive_event(transformed_event)
            elif hasattr(next_container, 'process_event'):
                await next_container.process_event(transformed_event)
            else:
                self.logger.warning(
                    "Next container doesn't support event reception",
                    container_id=stage.next_stage.container_id,
                    container_type=type(next_container).__name__
                )
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            stage.events_processed += 1
            stage.total_latency_ms += latency_ms
            
            self.stage_metrics[stage.stage_number]['events_processed'] += 1
            self.stage_metrics[stage.stage_number]['total_latency_ms'] += latency_ms
            self.stage_metrics[stage.stage_number]['average_latency_ms'] = (
                self.stage_metrics[stage.stage_number]['total_latency_ms'] / 
                self.stage_metrics[stage.stage_number]['events_processed']
            )
            
            self.metrics.events_sent += 1
            self.metrics.total_latency_ms += latency_ms
            
            self.logger.debug(
                "Pipeline event processed",
                stage_number=stage.stage_number,
                latency_ms=round(latency_ms, 2),
                correlation_id=correlation_id
            )
            
        except Exception as e:
            # Update error metrics
            stage.events_failed += 1
            self.stage_metrics[stage.stage_number]['events_failed'] += 1
            self.metrics.events_failed += 1
            self.metrics.errors_count += 1
            self.metrics.last_error_time = datetime.now()
            
            self.logger.error(
                "Pipeline event processing failed",
                stage_number=stage.stage_number,
                source_container=stage.container_id,
                target_container=stage.next_stage.container_id,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id
            )
            
            # Optionally implement error recovery here
            # For now, we'll stop propagation on error
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get detailed pipeline-specific metrics.
        
        Returns:
            Pipeline performance metrics
        """
        return {
            'total_stages': len(self.pipeline_stages),
            'stage_metrics': self.stage_metrics,
            'end_to_end_metrics': {
                'total_events': self.metrics.events_sent,
                'total_failures': self.metrics.events_failed,
                'average_latency_ms': self.metrics.average_latency_ms,
                'error_rate': self.metrics.error_rate
            },
            'pipeline_flow': [
                {
                    'stage': i,
                    'container': stage.container_id,
                    'events_processed': stage.events_processed,
                    'events_failed': stage.events_failed,
                    'average_latency_ms': (
                        stage.total_latency_ms / stage.events_processed 
                        if stage.events_processed > 0 else 0.0
                    )
                }
                for i, stage in enumerate(self.pipeline_stages)
            ]
        }
    
    def add_transformation_rule(self, from_type: EventType, to_type: EventType, transformer: Callable):
        """Add a custom event transformation rule.
        
        Args:
            from_type: Source event type
            to_type: Target event type  
            transformer: Transformation function
        """
        self.event_transformer.add_rule(from_type, to_type, transformer)
        
        self.logger.info(
            "Added transformation rule",
            from_type=from_type.name,
            to_type=to_type.name
        )