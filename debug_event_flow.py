"""Debug script to trace event flow through the pipeline."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from src.core.events.types import Event, EventType
from src.core.communication import AdapterFactory, create_adapter_with_logging
from src.core.communication.pipeline_adapter_protocol import PipelineAdapter

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patch the pipeline adapter to add detailed logging
original_route_event = PipelineAdapter.route_event

def debug_route_event(self, event: Event, source):
    logger.info(f"🔍 ROUTING: {event.event_type} from {source.name}")
    original_route_event(self, event, source)

PipelineAdapter.route_event = debug_route_event

# Also patch the forward handler creation
from src.core.communication import helpers

original_create_forward_handler = helpers.create_forward_handler

def debug_create_forward_handler(adapter, target):
    def forward_event(event: Event):
        logger.info(f"📤 FORWARDING: {event.event_type} to {target.name}")
        target.receive_event(event)
    return forward_event

helpers.create_forward_handler = debug_create_forward_handler

# Create test containers that log everything
class DebugContainer:
    def __init__(self, name: str, produces_events=None):
        self.name = name
        self.event_bus = DebugEventBus(self)
        self.received_count = 0
        self.produces_events = produces_events or []
        
    def receive_event(self, event: Event):
        self.received_count += 1
        logger.info(f"📥 {self.name} RECEIVED: {event.event_type} (total: {self.received_count})")
        
        # Simulate event transformation
        if self.name == "DataContainer" and event.event_type == EventType.START:
            self.publish_bar_event()
        elif self.name == "IndicatorContainer" and event.event_type == EventType.BAR:
            self.publish_indicator_event()
        elif self.name == "StrategyContainer" and event.event_type == EventType.INDICATOR:
            self.publish_signal_event()
            
    def publish_bar_event(self):
        event = Event(
            event_type=EventType.BAR,
            payload={"symbol": "TEST", "price": 100},
            timestamp=datetime.now(),
            source_id=self.name,
            container_id=self.name
        )
        logger.info(f"📢 {self.name} PUBLISHING: {event.event_type}")
        self.event_bus.publish(event)
        
    def publish_indicator_event(self):
        event = Event(
            event_type=EventType.INDICATOR,
            payload={"rsi": 50, "sma": 100},
            timestamp=datetime.now(),
            source_id=self.name,
            container_id=self.name
        )
        logger.info(f"📢 {self.name} PUBLISHING: {event.event_type}")
        self.event_bus.publish(event)
        
    def publish_signal_event(self):
        event = Event(
            event_type=EventType.SIGNAL,
            payload={"action": "BUY", "confidence": 0.8},
            timestamp=datetime.now(),
            source_id=self.name,
            container_id=self.name
        )
        logger.info(f"📢 {self.name} PUBLISHING: {event.event_type}")
        self.event_bus.publish(event)
        
    def on_output_event(self, handler):
        """Register output handler - this is what the containers use."""
        logger.info(f"🔗 {self.name} registering output handler")
        # Subscribe to the events this container produces
        for event_type in self.produces_events:
            self.event_bus.subscribe(event_type, handler)
        
    def publish_event(self, event: Event):
        self.event_bus.publish(event)
        
    def process(self, event: Event):
        return None


class DebugEventBus:
    def __init__(self, container):
        self.container = container
        self.handlers = {}
        
    def subscribe(self, event_type, handler):
        logger.info(f"🔔 {self.container.name} event bus: subscribing handler to {event_type}")
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def subscribe_all(self, handler):
        logger.info(f"🔔 {self.container.name} event bus: subscribing handler to ALL events")
        self.handlers['_all'] = handler
        
    def publish(self, event: Event):
        logger.info(f"📮 {self.container.name} event bus: publishing {event.event_type}")
        
        # Call specific handlers
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                logger.info(f"  → Calling handler for {event.event_type}")
                handler(event)
                
        # Call all handlers
        if '_all' in self.handlers:
            logger.info(f"  → Calling ALL handler")
            self.handlers['_all'](event)


def test_pipeline_flow():
    logger.info("\n" + "="*60)
    logger.info("TESTING PIPELINE EVENT FLOW")
    logger.info("="*60 + "\n")
    
    # Create containers with proper event types
    containers = {
        'DataContainer': DebugContainer('DataContainer', produces_events=[EventType.BAR]),
        'IndicatorContainer': DebugContainer('IndicatorContainer', produces_events=[EventType.INDICATOR]),
        'StrategyContainer': DebugContainer('StrategyContainer', produces_events=[EventType.SIGNAL]),
        'RiskContainer': DebugContainer('RiskContainer', produces_events=[EventType.ORDER])
    }
    
    # Create pipeline adapter
    factory = AdapterFactory()
    adapter = factory.create_adapter('test_pipeline', {
        'type': 'pipeline',
        'containers': ['DataContainer', 'IndicatorContainer', 'StrategyContainer', 'RiskContainer']
    })
    
    logger.info("\n🏗️  Setting up pipeline...")
    adapter.setup(containers)
    
    logger.info("\n🚀 Starting pipeline...")
    adapter.start()
    
    # Trigger initial event
    logger.info("\n⚡ Triggering initial START event...")
    start_event = Event(
        event_type=EventType.START,
        payload={},
        timestamp=datetime.now(),
        source_id='test',
        container_id='test'
    )
    
    containers['DataContainer'].receive_event(start_event)
    
    # Check results
    logger.info("\n📊 RESULTS:")
    for name, container in containers.items():
        logger.info(f"  {name}: received {container.received_count} events")
        
    # Verify the chain
    assert containers['IndicatorContainer'].received_count > 0, "IndicatorContainer should receive BAR events"
    assert containers['StrategyContainer'].received_count > 0, "StrategyContainer should receive INDICATOR events"
    
    logger.info("\n✅ Event flow test completed!")


if __name__ == "__main__":
    test_pipeline_flow()