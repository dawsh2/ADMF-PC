#!/usr/bin/env python3
"""
Test event flow through container hierarchy.
"""

import logging
from src.core.containers.container import Container, ContainerConfig
from src.core.events.types import Event, EventType
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create root container
    root_config = ContainerConfig(
        name="root",
        components=[],
        config={},
        container_type="root"
    )
    root = Container(root_config)
    
    # Create child containers
    child1_config = ContainerConfig(
        name="child1",
        components=[],
        config={},
        container_type="data"
    )
    child1 = Container(child1_config)
    root.add_child_container(child1)
    
    child2_config = ContainerConfig(
        name="child2", 
        components=[],
        config={},
        container_type="strategy"
    )
    child2 = Container(child2_config)
    root.add_child_container(child2)
    
    # Subscribe to events
    def log_event(bus_name):
        def handler(event):
            logger.info(f"{bus_name} received: {event.event_type} from {event.source_id}")
        return handler
    
    # Subscribe on all buses
    root.event_bus.subscribe(EventType.BAR.value, log_event("ROOT"))
    child1.event_bus.subscribe(EventType.BAR.value, log_event("CHILD1"))
    child2.event_bus.subscribe(EventType.BAR.value, log_event("CHILD2"))
    
    # Test 1: Child publishes locally
    logger.info("\n=== Test 1: Child1 publishes locally ===")
    test_event = Event(
        event_type=EventType.BAR.value,
        timestamp=datetime.now(),
        payload={"test": 1},
        source_id="child1_local"
    )
    child1.event_bus.publish(test_event)
    
    # Test 2: Child publishes to parent
    logger.info("\n=== Test 2: Child1 publishes to parent ===")
    test_event2 = Event(
        event_type=EventType.BAR.value,
        timestamp=datetime.now(),
        payload={"test": 2},
        source_id="child1_to_parent"
    )
    child1.publish_event(test_event2, target_scope="parent")
    
    # Test 3: Root publishes (should go to all children)
    logger.info("\n=== Test 3: Root publishes (auto-forwards to children) ===")
    test_event3 = Event(
        event_type=EventType.BAR.value,
        timestamp=datetime.now(),
        payload={"test": 3},
        source_id="root_broadcast"
    )
    root.publish_event(test_event3, target_scope="local")

if __name__ == "__main__":
    main()