#!/usr/bin/env python3
"""
Simpler test to verify event tracing setup.
"""

import asyncio
import logging

from src.core.events import EventBus, Event, EventType
from src.core.events.tracing import TracedEventBus, EventTracer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('src.core.events.tracing').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_direct_tracing():
    """Test event tracing directly."""
    
    logger.info("Testing direct event tracing")
    
    # Create event tracer
    tracer = EventTracer(correlation_id="test_001")
    
    # Create traced event bus
    bus = TracedEventBus("test_bus")
    logger.info(f"Bus tracer before set: {bus.tracer}")
    bus.set_tracer(tracer)
    logger.info(f"Bus tracer after set: {bus.tracer}")
    
    # Subscribe to events
    events_received = []
    
    def handler(event: Event):
        events_received.append(event)
        logger.info(f"Handler received: {event.event_type}")
    
    bus.subscribe(EventType.BAR, handler)
    
    # Publish some events
    for i in range(5):
        event = Event(
            event_type=EventType.BAR,
            payload={'bar_number': i},
            source_id='test_source'
        )
        bus.publish(event)
    
    # Get trace summary
    summary = tracer.get_summary()
    logger.info(f"🔍 Trace Summary:")
    logger.info(f"  Total events: {summary.get('total_events', 0)}")
    logger.info(f"  Event types: {summary.get('event_types', {})}")
    logger.info(f"  Source containers: {list(summary.get('source_containers', {}).keys())}")
    logger.info(f"  Events received by handler: {len(events_received)}")
    
    # Verify events were traced
    if summary['total_events'] > 0:
        logger.info("✅ Event tracing is working correctly!")
    else:
        logger.error("❌ No events were traced")


if __name__ == "__main__":
    asyncio.run(test_direct_tracing())