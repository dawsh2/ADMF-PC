"""
Test the Event Router integration with composable containers.

This test demonstrates how the Event Router enables cross-container
communication while maintaining isolation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.containers.composable import BaseComposableContainer, ContainerRole
from src.core.events.routing import (
    EventRouter, EventPublication, EventSubscription, EventScope
)
from src.core.events.types import Event, EventType
from src.core.logging.structured import StructuredLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestIndicatorContainer(BaseComposableContainer):
    """Test indicator container that publishes indicators."""
    
    def __init__(self):
        super().__init__(
            role=ContainerRole.INDICATOR,
            name="test_indicator",
            config={
                "events": {
                    "publishes": [
                        {
                            "events": ["INDICATORS"],
                            "scope": "GLOBAL"  # Publish globally
                        }
                    ]
                }
            }
        )
        self.published_count = 0
    
    async def publish_test_indicator(self):
        """Publish a test indicator event."""
        event = Event(
            event_type=EventType.INDICATORS,
            payload={
                "RSI": 45.5,
                "SMA_20": 150.25
            },
            metadata={"source": "test_indicator"}
        )
        
        # Use routed event publishing
        self.publish_routed_event(event, EventScope.GLOBAL)
        self.published_count += 1
        logger.info(f"Published indicator event #{self.published_count}")


class TestStrategyContainer(BaseComposableContainer):
    """Test strategy container that subscribes to indicators."""
    
    def __init__(self, indicator_container_id: str):
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="test_strategy",
            config={
                "events": {
                    "subscribes_to": [
                        {
                            "source": indicator_container_id,  # Use actual container ID
                            "events": ["INDICATORS"]
                        }
                    ]
                }
            }
        )
        self.received_indicators = []
        
    def handle_routed_event(self, event: Event, source: str):
        """Handle routed indicator events."""
        if event.event_type == EventType.INDICATORS:
            self.received_indicators.append(event.payload)
            logger.info(f"Strategy received indicators from {source}: {event.payload}")


async def test_event_router():
    """Test the event router with indicator -> strategy flow."""
    
    logger.info("Starting Event Router integration test")
    
    # Create event router
    router = EventRouter(enable_debugging=True)
    
    # Create containers
    indicator_container = TestIndicatorContainer()
    strategy_container = TestStrategyContainer(indicator_container.metadata.container_id)
    
    # Initialize containers
    await indicator_container.initialize()
    await strategy_container.initialize()
    
    # Register containers with router
    indicator_container.register_with_router(router)
    strategy_container.register_with_router(router)
    
    # Set up container hierarchy (they're siblings in this test)
    router.set_container_hierarchy({
        indicator_container.metadata.container_id: {
            'parent_id': None,
            'children_ids': []
        },
        strategy_container.metadata.container_id: {
            'parent_id': None,
            'children_ids': []
        }
    })
    
    # Validate topology
    validation = router.validate_topology()
    logger.info(f"Topology validation: {validation.is_valid}")
    if validation.warnings:
        logger.warning(f"Validation warnings: {validation.warnings}")
    
    # Start containers
    await indicator_container.start()
    await strategy_container.start()
    
    # Publish some test indicators
    for i in range(3):
        await indicator_container.publish_test_indicator()
        await asyncio.sleep(0.1)  # Small delay to ensure processing
    
    # Check results
    logger.info(f"Indicator container published: {indicator_container.published_count} events")
    logger.info(f"Strategy container received: {len(strategy_container.received_indicators)} events")
    
    # Print received data
    for i, data in enumerate(strategy_container.received_indicators):
        logger.info(f"  Received indicator #{i+1}: {data}")
    
    # Get router metrics
    metrics = router.get_metrics()
    logger.info(f"Router metrics: {metrics}")
    
    # Validate results
    assert indicator_container.published_count == 3, "Should have published 3 events"
    assert len(strategy_container.received_indicators) == 3, "Should have received 3 events"
    
    # Clean up
    await indicator_container.stop()
    await strategy_container.stop()
    await indicator_container.dispose()
    await strategy_container.dispose()
    
    logger.info("Event Router integration test completed successfully!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_event_router())