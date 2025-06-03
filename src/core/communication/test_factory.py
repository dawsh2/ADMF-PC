"""Test the communication factory and layer functionality."""

import asyncio
from datetime import datetime

from .factory import EventCommunicationFactory, CommunicationLayer
from .pipeline_adapter import PipelineCommunicationAdapter
from ..logging.log_manager import LogManager
from ..events.types import Event, EventType


class MockContainer:
    """Mock container for testing."""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.received_events = []
        self._output_handlers = []
        
    def on_output_event(self, handler):
        """Register output handler."""
        self._output_handlers.append(handler)
        
    async def receive_event(self, event: Event):
        """Receive an event."""
        self.received_events.append(event)
        print(f"{self.container_id} received event: {event.event_type}")
        
        # Forward to output handlers
        for handler in self._output_handlers:
            handler(event)


async def test_factory():
    """Test the communication factory."""
    print("Testing Communication Factory\n")
    
    # Create log manager
    log_config = {
        'base_log_dir': 'test_logs',
        'retention_policy': {'max_age_days': 1}
    }
    log_manager = LogManager(coordinator_id='test_coordinator', config=log_config)
    
    # Create factory
    factory = EventCommunicationFactory('test_coordinator', log_manager)
    print(f"Factory created with adapter types: {list(factory.adapter_registry.keys())}\n")
    
    # Create mock containers
    containers = {
        'data': MockContainer('data_container'),
        'strategy': MockContainer('strategy_container'),
        'risk': MockContainer('risk_container'),
        'execution': MockContainer('execution_container')
    }
    
    # Create communication configuration
    comm_config = {
        'adapters': [
            {
                'type': 'pipeline',
                'name': 'main_pipeline',
                'containers': ['data', 'strategy', 'risk', 'execution']
            }
        ]
    }
    
    # Create communication layer
    comm_layer = factory.create_communication_layer(comm_config, containers)
    print(f"Communication layer created with {len(comm_layer.adapters)} adapters")
    
    # Setup all adapters
    await comm_layer.setup_all_adapters()
    print("All adapters setup complete\n")
    
    # Get system metrics before processing
    metrics_before = comm_layer.get_system_metrics()
    print(f"Initial metrics: {metrics_before['total_events']} events processed")
    print(f"Health status: {metrics_before['overall_health']}\n")
    
    # Send test events through pipeline
    print("Sending test events through pipeline...")
    data_container = containers['data']
    
    for i in range(5):
        event = Event(
            event_type=EventType.BAR,
            payload={'symbol': 'AAPL', 'price': 150.0 + i, 'index': i},
            timestamp=datetime.now(),
            source_id='test_source',
            container_id='data_container',
            metadata={'test_run': True}
        )
        
        # Trigger pipeline by emitting from data container
        for handler in data_container._output_handlers:
            handler(event)
        
        # Allow async processing
        await asyncio.sleep(0.1)
    
    print(f"\nEvents received by containers:")
    for name, container in containers.items():
        print(f"  {name}: {len(container.received_events)} events")
    
    # Get pipeline-specific metrics
    pipeline_adapter = comm_layer.get_adapter('main_pipeline')
    if isinstance(pipeline_adapter, PipelineCommunicationAdapter):
        pipeline_metrics = pipeline_adapter.get_pipeline_metrics()
        print(f"\nPipeline metrics:")
        print(f"  Total stages: {pipeline_metrics['total_stages']}")
        print(f"  Events processed: {pipeline_metrics['end_to_end_metrics']['total_events']}")
        print(f"  Average latency: {pipeline_metrics['end_to_end_metrics']['average_latency_ms']:.2f}ms")
    
    # Get final system metrics
    metrics_after = comm_layer.get_system_metrics()
    print(f"\nFinal system metrics:")
    print(f"  Total events: {metrics_after['total_events']}")
    print(f"  Events per second: {metrics_after['events_per_second']:.2f}")
    print(f"  Overall health: {metrics_after['overall_health']}")
    
    # Get adapter status summary
    status_summary = comm_layer.get_adapter_status_summary()
    print(f"\nAdapter status summary: {status_summary}")
    
    # Cleanup
    print("\nCleaning up...")
    await comm_layer.cleanup()
    await factory.cleanup_all_adapters()
    print("Cleanup complete")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_factory())