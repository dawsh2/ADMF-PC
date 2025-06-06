#!/usr/bin/env python3
"""
Simple test to verify event flow between containers.
"""

import asyncio
import logging
from datetime import datetime
from src.core.types.events import Event, EventType
from src.core.types.trading import Bar

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_simple_event_flow():
    """Test simple event flow between two containers."""
    
    from src.core.containers.container import Container, ContainerConfig
    from src.core.containers.protocols import ContainerRole
    from src.core.communication.factory import AdapterFactory
    
    # Create two simple containers
    source_container = Container(ContainerConfig(
        role=ContainerRole.DATA,
        name='source',
        container_id='source',
        config={},
        capabilities=set()
    ))
    
    target_container = Container(ContainerConfig(
        role=ContainerRole.PORTFOLIO,
        name='target',
        container_id='target',
        config={},
        capabilities=set()
    ))
    
    # Track received events
    events_received = []
    
    def handle_event(event: Event):
        logger.info(f"Target received event: {event.event_type}, payload keys: {list(event.payload.keys())}")
        events_received.append(event)
    
    # Subscribe target to FEATURES events
    target_container.event_bus.subscribe(EventType.FEATURES, handle_event)
    
    # Create broadcast adapter
    adapter_factory = AdapterFactory()
    broadcast = adapter_factory.create_adapter(
        name='test_broadcast',
        config={
            'type': 'broadcast',
            'source': 'source',
            'targets': ['target']
        }
    )
    
    # Wire containers
    containers = {
        'source': source_container,
        'target': target_container
    }
    broadcast.setup(containers)
    broadcast.start()
    
    # Initialize containers
    await source_container.initialize()
    await target_container.initialize()
    await source_container.start()
    await target_container.start()
    
    # Publish test FEATURES event from source
    logger.info("Publishing FEATURES event from source")
    source_container.event_bus.publish(Event(
        event_type=EventType.FEATURES,
        payload={
            'symbol': 'TEST',
            'features': {'sma_20': 100.0, 'rsi': 50.0},
            'bar': {'close': 100.0}
        },
        source_id='source'
    ))
    
    # Wait a bit for event propagation
    await asyncio.sleep(0.1)
    
    # Check results
    logger.info(f"Events received by target: {len(events_received)}")
    for event in events_received:
        logger.info(f"Event type: {event.event_type}, payload: {event.payload}")
    
    # Cleanup
    await source_container.stop()
    await target_container.stop()
    broadcast.stop()
    
    return len(events_received) > 0


if __name__ == "__main__":
    success = asyncio.run(test_simple_event_flow())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")