"""Example of integrating the Communication Factory with WorkflowCoordinator.

This example shows how to:
1. Setup communication adapters through the coordinator
2. Configure pipeline-based event flow
3. Monitor communication metrics
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from ..coordinator.coordinator import WorkflowCoordinator
from ..containers.enhanced_container import EnhancedContainer
from ..events.types import Event, EventType
from .factory import EventCommunicationFactory, CommunicationLayer


class CommunicationEnabledCoordinator(WorkflowCoordinator):
    """Extended coordinator with communication layer support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize coordinator with communication support.
        
        Args:
            config: Coordinator configuration
        """
        super().__init__(config)
        
        # Initialize communication factory
        self.communication_factory = EventCommunicationFactory(
            self.coordinator_id,
            self.log_manager
        )
        self.communication_layer: Optional[CommunicationLayer] = None
        
        self.logger.info(
            "Initialized coordinator with communication support",
            coordinator_id=self.coordinator_id,
            lifecycle_operation="coordinator_initialization"
        )
    
    async def setup_communication(self, communication_config: Dict[str, Any]):
        """Setup event communication system.
        
        Args:
            communication_config: Communication configuration with adapter definitions
        """
        self.logger.info(
            "Initializing event communication system",
            communication_pattern=communication_config.get('pattern', 'default'),
            adapter_count=len(communication_config.get('adapters', [])),
            lifecycle_operation="communication_initialization"
        )
        
        try:
            # Create communication layer with all registered containers
            self.communication_layer = self.communication_factory.create_communication_layer(
                communication_config,
                self.containers
            )
            
            # Setup all adapters
            await self.communication_layer.setup_all_adapters()
            
            self.logger.info(
                "Event communication system ready",
                active_adapters=len(self.communication_layer.adapters),
                lifecycle_operation="communication_ready"
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize communication system",
                error=str(e),
                error_type=type(e).__name__,
                lifecycle_operation="communication_initialization_error"
            )
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status including communication metrics.
        
        Returns:
            System status with communication information
        """
        # Get base status from parent
        status = await super().get_system_status()
        
        # Add communication status
        if self.communication_layer:
            try:
                communication_metrics = self.communication_layer.get_system_metrics()
                status['communication'] = communication_metrics
                
            except Exception as e:
                self.logger.error(
                    "Error getting communication status",
                    error=str(e),
                    error_type=type(e).__name__,
                    lifecycle_operation="status_collection_error"
                )
                status['communication'] = {"status": "error", "error": str(e)}
        else:
            status['communication'] = {"status": "not_initialized"}
        
        return status
    
    async def shutdown(self):
        """Enhanced shutdown with communication cleanup."""
        self.logger.info(
            "Starting coordinator shutdown with communication cleanup",
            lifecycle_operation="coordinator_shutdown"
        )
        
        # Cleanup communication layer first
        if self.communication_layer:
            await self.communication_layer.cleanup()
        
        # Cleanup communication factory
        await self.communication_factory.cleanup_all_adapters()
        
        # Continue with parent shutdown
        await super().shutdown()


async def example_pipeline_communication():
    """Example: Setup a simple pipeline communication pattern."""
    
    print("=== Pipeline Communication Example ===\n")
    
    # Create coordinator configuration
    coordinator_config = {
        'coordinator_id': 'pipeline_example',
        'log_dir': 'example_logs',
        'logging': {
            'retention_policy': {'max_age_days': 7},
            'performance': {'async_writing': True}
        }
    }
    
    # Create coordinator
    coordinator = CommunicationEnabledCoordinator(coordinator_config)
    
    # Create containers
    containers_config = {
        'data_container': {'type': 'data', 'expected_input_type': EventType.BAR},
        'indicator_container': {'type': 'indicator', 'expected_input_type': EventType.BAR},
        'strategy_container': {'type': 'strategy', 'expected_input_type': EventType.INDICATOR},
        'risk_container': {'type': 'risk', 'expected_input_type': EventType.SIGNAL},
        'execution_container': {'type': 'execution', 'expected_input_type': EventType.ORDER}
    }
    
    for container_id, config in containers_config.items():
        container = EnhancedContainer(container_id, config['type'])
        if 'expected_input_type' in config:
            container.expected_input_type = config['expected_input_type']
        coordinator.containers[container_id] = container
    
    print(f"Created {len(coordinator.containers)} containers\n")
    
    # Setup pipeline communication
    communication_config = {
        'pattern': 'pipeline',
        'adapters': [
            {
                'type': 'pipeline',
                'name': 'main_trading_pipeline',
                'containers': [
                    'data_container',
                    'indicator_container',
                    'strategy_container',
                    'risk_container',
                    'execution_container'
                ]
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    print("Communication layer setup complete\n")
    
    # Simulate event flow
    print("Simulating trading events...")
    data_container = coordinator.containers['data_container']
    
    for i in range(3):
        # Create market data event
        bar_event = Event(
            event_type=EventType.BAR,
            payload={
                'symbol': 'AAPL',
                'timestamp': datetime.now(),
                'open': 150.0 + i,
                'high': 152.0 + i,
                'low': 149.0 + i,
                'close': 151.0 + i,
                'volume': 1000000
            },
            timestamp=datetime.now(),
            source_id='market_data_feed',
            container_id='data_container',
            metadata={'bar_number': i}
        )
        
        # Emit event to pipeline
        data_container.emit_output_event(bar_event)
        
        # Allow processing
        await asyncio.sleep(0.1)
    
    print("\n")
    
    # Get communication metrics
    status = await coordinator.get_system_status()
    comm_metrics = status['communication']
    
    print("Communication Metrics:")
    print(f"  Total adapters: {comm_metrics['total_adapters']}")
    print(f"  Active adapters: {comm_metrics['active_adapters']}")
    print(f"  Total events: {comm_metrics['total_events']}")
    print(f"  Events per second: {comm_metrics['events_per_second']:.2f}")
    print(f"  Overall health: {comm_metrics['overall_health']}")
    
    # Get pipeline-specific metrics
    pipeline_adapter = coordinator.communication_layer.get_adapter('main_trading_pipeline')
    if pipeline_adapter:
        pipeline_metrics = pipeline_adapter.get_pipeline_metrics()
        print("\nPipeline Performance:")
        print(f"  Stages: {pipeline_metrics['total_stages']}")
        print(f"  End-to-end latency: {pipeline_metrics['end_to_end_metrics']['average_latency_ms']:.2f}ms")
        
        print("\nPer-Stage Metrics:")
        for stage in pipeline_metrics['pipeline_flow']:
            print(f"  Stage {stage['stage']} ({stage['container']}): "
                  f"{stage['events_processed']} events, "
                  f"{stage['average_latency_ms']:.2f}ms avg latency")
    
    # Cleanup
    print("\nShutting down...")
    await coordinator.shutdown()
    print("Example complete!")


async def example_multi_strategy_fix():
    """Example: Fix circular dependency in multi-strategy backtest."""
    
    print("\n=== Multi-Strategy Circular Dependency Fix ===\n")
    
    # Create coordinator
    coordinator_config = {
        'coordinator_id': 'multi_strategy_fix',
        'log_dir': 'example_logs'
    }
    
    coordinator = CommunicationEnabledCoordinator(coordinator_config)
    
    # Create containers (simulating the problematic hierarchy)
    containers = [
        ('classifier_container', 'classifier'),
        ('risk_container', 'risk'),
        ('portfolio_container', 'portfolio'),
        ('strategy_container', 'strategy'),
        ('execution_container', 'execution'),
        ('data_container', 'data'),
        ('indicator_container', 'indicator')
    ]
    
    for container_id, container_type in containers:
        container = EnhancedContainer(container_id, container_type)
        coordinator.containers[container_id] = container
    
    # Fix with pipeline adapter (no circular dependencies)
    communication_config = {
        'pattern': 'linear_pipeline',
        'adapters': [
            {
                'type': 'pipeline',
                'name': 'fixed_multi_strategy_flow',
                'containers': [
                    'data_container',
                    'indicator_container',
                    'classifier_container',
                    'strategy_container',
                    'risk_container',
                    'portfolio_container',
                    'execution_container'
                ]
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    print("Fixed configuration:")
    print("  - Linear pipeline eliminates circular dependencies")
    print("  - Events flow in one direction only")
    print("  - Container hierarchy preserved for configuration")
    print("  - Event flow decoupled from hierarchy")
    
    # Cleanup
    await coordinator.shutdown()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_pipeline_communication())
    asyncio.run(example_multi_strategy_fix())