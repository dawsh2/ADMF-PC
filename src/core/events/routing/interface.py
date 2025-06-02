"""
Event interface for containers to participate in cross-container routing.

This module provides a mixin that containers can use to integrate with
the event routing system while maintaining their isolated event buses.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .protocols import (
    EventRouterProtocol, EventPublication, EventSubscription,
    EventScope, EventType
)
from ..types import Event


logger = logging.getLogger(__name__)


class ContainerEventInterface:
    """
    Mixin for containers to participate in event routing.
    
    This interface allows containers to:
    - Declare what events they publish
    - Subscribe to events from other containers
    - Maintain isolation while enabling cross-container communication
    
    Usage:
        class MyContainer(BaseComposableContainer, ContainerEventInterface):
            def __init__(self):
                BaseComposableContainer.__init__(self, ...)
                ContainerEventInterface.__init__(self)
                
                # Declare publications
                self.declare_publications([
                    EventPublication(
                        events={EventType.SIGNAL},
                        scope=EventScope.PARENT
                    )
                ])
                
                # Declare subscriptions
                self.declare_subscriptions([
                    EventSubscription(
                        source="indicator_container",
                        events={EventType.INDICATORS}
                    )
                ])
    """
    
    def __init__(self):
        """Initialize event interface."""
        self._event_router: Optional[EventRouterProtocol] = None
        self._publications: List[EventPublication] = []
        self._subscriptions: List[EventSubscription] = []
        self._routed_event_handlers = {}
        
    def register_with_router(self, router: EventRouterProtocol) -> None:
        """
        Register this container with the event router.
        
        This method should be called after the container is initialized
        and its publications/subscriptions are declared.
        
        Args:
            router: The event router to register with
        """
        self._event_router = router
        
        # Register publications
        if self._publications:
            router.register_publisher(
                self.metadata.container_id, 
                self._publications
            )
            logger.info(
                f"Container {self.metadata.name} registered "
                f"{len(self._publications)} publications"
            )
            
        # Register subscriptions
        if self._subscriptions:
            router.register_subscriber(
                self.metadata.container_id,
                self._subscriptions,
                self.handle_routed_event
            )
            logger.info(
                f"Container {self.metadata.name} registered "
                f"{len(self._subscriptions)} subscriptions"
            )
    
    def unregister_from_router(self) -> None:
        """Unregister from event router during cleanup."""
        if self._event_router:
            self._event_router.unregister_container(self.metadata.container_id)
            self._event_router = None
    
    def publish_routed_event(
        self, 
        event: Event, 
        scope: Optional[EventScope] = None
    ) -> None:
        """
        Publish event through router to other containers.
        
        This method publishes the event both locally (to the container's
        own event bus) and through the router to other containers based
        on the configured scope.
        
        Args:
            event: The event to publish
            scope: Override the default scope for this event
        """
        # Always publish locally first
        self.event_bus.publish(event)
        
        # Then route to other containers if router is available
        if self._event_router:
            self._event_router.route_event(
                self.metadata.container_id,
                event,
                scope
            )
        elif scope and scope != EventScope.LOCAL:
            logger.warning(
                f"Container {self.metadata.name} tried to publish "
                f"event {event.event_type} with scope {scope} but "
                "no router is registered"
            )
    
    def handle_routed_event(self, event: Event, source: str) -> None:
        """
        Handle event routed from another container.
        
        Default implementation processes the event through the normal
        event handling pipeline. Containers can override this to handle
        routed events differently.
        
        Args:
            event: The routed event
            source: ID of the container that published the event
        """
        # Add source metadata if not present
        if 'source_container' not in event.metadata:
            event.metadata['source_container'] = source
        
        # Check for custom handlers
        event_type_str = str(event.event_type)
        if event_type_str in self._routed_event_handlers:
            handler = self._routed_event_handlers[event_type_str]
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event, source))
                else:
                    handler(event, source)
            except Exception as e:
                logger.error(
                    f"Error in routed event handler for {event_type_str}: {e}",
                    exc_info=True
                )
        else:
            # Default: process through normal event handling
            if hasattr(self, 'process_event'):
                try:
                    # Use asyncio.create_task for async processing
                    if asyncio.iscoroutinefunction(self.process_event):
                        asyncio.create_task(self.process_event(event))
                    else:
                        self.process_event(event)
                except Exception as e:
                    logger.error(
                        f"Error processing routed event {event.event_type}: {e}",
                        exc_info=True
                    )
    
    def declare_publications(self, publications: List[EventPublication]) -> None:
        """
        Declare what events this container publishes.
        
        Should be called during container initialization before
        registering with the router.
        
        Args:
            publications: List of event publication declarations
        """
        self._publications = publications
        
    def declare_subscriptions(self, subscriptions: List[EventSubscription]) -> None:
        """
        Declare what events this container subscribes to.
        
        Should be called during container initialization before
        registering with the router.
        
        Args:
            subscriptions: List of event subscription declarations
        """
        self._subscriptions = subscriptions
    
    def add_routed_event_handler(
        self, 
        event_type: Union[EventType, str],
        handler: callable
    ) -> None:
        """
        Add a custom handler for specific routed event types.
        
        This allows containers to handle routed events differently
        from local events if needed.
        
        Args:
            event_type: The event type to handle
            handler: Function to call for this event type
        """
        event_type_str = str(event_type)
        self._routed_event_handlers[event_type_str] = handler
    
    def configure_event_routing(self, config: Dict[str, Any]) -> None:
        """
        Configure event routing from configuration dict.
        
        This method parses YAML configuration and sets up
        publications and subscriptions accordingly.
        
        Args:
            config: Event routing configuration from YAML
        """
        # Parse publications
        if 'publishes' in config:
            publications = []
            for pub_config in config['publishes']:
                # Convert event type strings to EventType if needed
                events = set()
                for event in pub_config.get('events', []):
                    if isinstance(event, str):
                        # Try to convert to EventType
                        try:
                            events.add(EventType[event])
                        except (KeyError, AttributeError):
                            # Keep as string if not a valid EventType
                            events.add(event)
                    else:
                        events.add(event)
                
                # Create publication
                pub = EventPublication(
                    events=events,
                    scope=EventScope[pub_config.get('scope', 'PARENT').upper()],
                    qos=pub_config.get('qos', 'best_effort'),
                    priority=pub_config.get('priority', 0)
                )
                publications.append(pub)
            
            self.declare_publications(publications)
        
        # Parse subscriptions
        if 'subscribes_to' in config:
            subscriptions = []
            for sub_config in config['subscribes_to']:
                # Convert event type strings
                events = set()
                for event in sub_config.get('events', []):
                    if isinstance(event, str):
                        try:
                            events.add(EventType[event])
                        except (KeyError, AttributeError):
                            events.add(event)
                    else:
                        events.add(event)
                
                # Create subscription
                sub = EventSubscription(
                    source=sub_config['source'],
                    events=events,
                    transform=sub_config.get('transform'),
                    # Add batching config if present
                    batching=sub_config.get('batching')
                )
                subscriptions.append(sub)
            
            self.declare_subscriptions(subscriptions)
    
    def get_event_routing_info(self) -> Dict[str, Any]:
        """
        Get information about this container's event routing.
        
        Useful for debugging and visualization.
        
        Returns:
            Dictionary with publications and subscriptions info
        """
        return {
            "container_id": self.metadata.container_id,
            "container_name": self.metadata.name,
            "publishes": [
                {
                    "events": [str(e) for e in pub.events],
                    "scope": pub.scope.value,
                    "qos": pub.qos.value
                }
                for pub in self._publications
            ],
            "subscribes_to": [
                {
                    "source": sub.source,
                    "events": [str(e) for e in sub.events],
                    "has_filter": sub.filters is not None,
                    "has_transform": sub.transform is not None
                }
                for sub in self._subscriptions
            ]
        }