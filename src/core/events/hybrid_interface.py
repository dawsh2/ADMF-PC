"""
Hybrid Container Interface for mixed Event Router + Direct Bus communication.

This module provides the base interface for containers that support both:
1. External communication via Tiered Event Router (cross-container)
2. Internal communication via Direct Event Bus (sub-container)
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .types import Event, EventType
from .routing.protocols import (
    EventRouterProtocol, EventPublication, EventSubscription, 
    EventScope, EventQoS, EventFilter
)


logger = logging.getLogger(__name__)


class CommunicationTier(Enum):
    """Performance tiers for Event Router communication"""
    FAST = "fast"          # < 1ms - BAR, TICK, QUOTE
    STANDARD = "standard"  # < 10ms - SIGNAL, INDICATOR  
    RELIABLE = "reliable"  # 100% delivery - ORDER, FILL


@dataclass
class ExternalEventConfig:
    """Configuration for external Event Router communication"""
    publications: List[EventPublication]
    subscriptions: List[EventSubscription]


class EventBus:
    """Simple internal event bus for sub-container communication"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.all_subscribers: List[Callable] = []
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to specific event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable) -> None:
        """Subscribe to all events"""
        self.all_subscribers.append(callback)
    
    def publish(self, event: Event) -> None:
        """Publish event to subscribers"""
        # Notify specific type subscribers
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(event))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}", exc_info=True)
        
        # Notify all-event subscribers
        for callback in self.all_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in all-event subscriber: {e}", exc_info=True)


class HybridContainerInterface:
    """
    Container interface supporting hybrid communication patterns.
    
    External Communication: Tiered Event Router for cross-container
    Internal Communication: Direct Event Bus for sub-containers
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        
        # External communication via Tiered Event Router
        self.external_router: Optional['TieredEventRouter'] = None
        self._external_publications: List[EventPublication] = []
        self._external_subscriptions: List[EventSubscription] = []
        
        # Internal communication via Direct Event Bus
        self.internal_bus = EventBus()
        self.children: List['HybridContainerInterface'] = []
        self.parent: Optional['HybridContainerInterface'] = None
        
        # Event tier mapping for automatic tier selection
        self.event_tier_map = {
            EventType.BAR: CommunicationTier.FAST,
            EventType.TICK: CommunicationTier.FAST,
            EventType.QUOTE: CommunicationTier.FAST,
            EventType.SIGNAL: CommunicationTier.STANDARD,
            EventType.INDICATORS: CommunicationTier.STANDARD,
            EventType.PORTFOLIO: CommunicationTier.STANDARD,
            EventType.ORDER: CommunicationTier.RELIABLE,
            EventType.FILL: CommunicationTier.RELIABLE,
            EventType.SYSTEM: CommunicationTier.RELIABLE
        }
    
    # === External Communication (Cross-Container) ===
    
    def register_with_router(self, router: 'TieredEventRouter') -> None:
        """Register for cross-container communication via Event Router"""
        self.external_router = router
        
        # Register publications
        if self._external_publications:
            router.register_publisher(self.container_id, self._external_publications)
            logger.info(f"📡 {self.container_id} registered {len(self._external_publications)} publications")
        
        # Register subscriptions  
        if self._external_subscriptions:
            router.register_subscriber(
                self.container_id,
                self._external_subscriptions,
                self.handle_external_event
            )
            logger.info(f"📡 {self.container_id} registered {len(self._external_subscriptions)} subscriptions")
        
        # CASCADE: Register all children for external communication
        for child in self.children:
            if hasattr(child, 'register_with_router'):
                child.register_with_router(router)
                logger.debug(f"📡 Cascaded router registration to {child.container_id}")
    
    def publish_external(self, event: Event, tier: Optional[CommunicationTier] = None, scope: Optional[EventScope] = None) -> None:
        """Publish event to other containers via Event Router"""
        if not self.external_router:
            logger.warning(f"Container {self.container_id} not registered with router - cannot publish external event")
            return
        
        # Auto-determine tier if not specified
        if tier is None:
            tier = self.event_tier_map.get(event.event_type, CommunicationTier.STANDARD)
        
        # Use the correct EventRouterProtocol interface
        if hasattr(self.external_router, 'route_event_with_tier'):
            # Our TieredEventRouter with tier support
            self.external_router.route_event_with_tier(event, self.container_id, tier.value, scope)
        else:
            # Standard EventRouterProtocol interface
            self.external_router.route_event(self.container_id, event, scope)
        
        logger.debug(f"📡 {self.container_id} published {event.event_type} via {tier.value if tier else 'auto'} tier")
    
    def handle_external_event(self, event: Event, source: str) -> None:
        """Handle events routed from other containers"""
        # Add source metadata
        if not hasattr(event, 'metadata'):
            event.metadata = {}
        event.metadata['source_container'] = source
        event.metadata['communication_type'] = 'external'
        
        logger.debug(f"📨 {self.container_id} received {event.event_type} from {source} (external)")
        
        # Process through normal event handling
        if hasattr(self, 'process_event'):
            try:
                if asyncio.iscoroutinefunction(self.process_event):
                    asyncio.create_task(self.process_event(event))
                else:
                    self.process_event(event)
            except Exception as e:
                logger.error(f"Error processing external event {event.event_type} in {self.container_id}: {e}", exc_info=True)
    
    # === Internal Communication (Sub-Container) ===
    
    def add_child_container(self, child: 'HybridContainerInterface') -> None:
        """Add child with automatic communication setup"""
        self.children.append(child)
        child.parent = self
        
        # Setup internal event bridging
        child.internal_bus.subscribe_all(self._handle_child_event)
        self.internal_bus.subscribe_all(child._handle_parent_event)
        
        # Register child with external router if available
        if self.external_router and hasattr(child, 'register_with_router'):
            child.register_with_router(self.external_router)
        
        logger.info(f"📨 Added child {child.container_id} to {self.container_id} (internal communication setup)")
    
    def publish_internal(self, event: Event, scope: str = "children") -> None:
        """Publish event within container boundary via direct event bus"""
        # Add metadata
        if not hasattr(event, 'metadata'):
            event.metadata = {}
        event.metadata['communication_type'] = 'internal'
        event.metadata['internal_scope'] = scope
        
        if scope == "children":
            for child in self.children:
                child.internal_bus.publish(event)
                logger.debug(f"📨 {self.container_id} → {child.container_id} (internal)")
        
        elif scope == "parent":
            if self.parent:
                self.parent.internal_bus.publish(event)
                logger.debug(f"📨 {self.container_id} → {self.parent.container_id} (internal)")
        
        elif scope == "siblings":
            if self.parent:
                for sibling in self.parent.children:
                    if sibling != self:
                        sibling.internal_bus.publish(event)
                        logger.debug(f"📨 {self.container_id} → {sibling.container_id} (internal)")
        
        else:
            logger.warning(f"Unknown internal scope: {scope}")
    
    def _handle_child_event(self, event: Event) -> None:
        """Handle events from child containers"""
        # Check if this event should be forwarded externally
        flattened_types = set()
        for pub in self._external_publications:
            flattened_types.update(pub.events)
        
        if event.event_type in flattened_types:
            # Forward to external containers
            self.publish_external(event)
            logger.debug(f"🔄 {self.container_id} forwarded {event.event_type} from child externally")
        
        # Also process internally if we handle this event type
        if hasattr(self, 'process_event'):
            try:
                if asyncio.iscoroutinefunction(self.process_event):
                    asyncio.create_task(self.process_event(event))
                else:
                    self.process_event(event)
            except Exception as e:
                logger.error(f"Error processing child event {event.event_type} in {self.container_id}: {e}", exc_info=True)
    
    def _handle_parent_event(self, event: Event) -> None:
        """Handle events from parent container"""
        logger.debug(f"📨 {self.container_id} received {event.event_type} from parent (internal)")
        
        if hasattr(self, 'process_event'):
            try:
                if asyncio.iscoroutinefunction(self.process_event):
                    asyncio.create_task(self.process_event(event))
                else:
                    self.process_event(event)
            except Exception as e:
                logger.error(f"Error processing parent event {event.event_type} in {self.container_id}: {e}", exc_info=True)
    
    # === Configuration Support ===
    
    def configure_external_communication(self, config: Dict[str, Any]) -> None:
        """Configure external Event Router communication from config"""
        if 'external_events' not in config:
            return
        
        ext_config = config['external_events']
        
        # Configure publications
        if 'publishes' in ext_config:
            publications = []
            for pub_config in ext_config['publishes']:
                # Convert string event types to EventType enums
                events = set()
                for event_name in pub_config['events']:
                    if isinstance(event_name, str):
                        try:
                            events.add(EventType[event_name])
                        except KeyError:
                            logger.warning(f"Unknown event type: {event_name}")
                    else:
                        events.add(event_name)
                
                pub = EventPublication(
                    events=events,
                    scope=EventScope[pub_config.get('scope', 'GLOBAL').upper()],
                    qos=EventQoS[pub_config.get('qos', 'BEST_EFFORT').upper()],
                    priority=pub_config.get('priority', 0)
                )
                # Add tier information as metadata
                pub.metadata['tier'] = pub_config.get('tier', 'standard')
                publications.append(pub)
            
            self._external_publications = publications
        
        # Configure subscriptions
        if 'subscribes' in ext_config:
            subscriptions = []
            for sub_config in ext_config['subscribes']:
                # Convert string event types to EventType enums
                events = set()
                for event_name in sub_config['events']:
                    if isinstance(event_name, str):
                        try:
                            events.add(EventType[event_name])
                        except KeyError:
                            logger.warning(f"Unknown event type: {event_name}")
                    else:
                        events.add(event_name)
                
                # Create EventFilter if filters are provided
                event_filter = None
                if sub_config.get('filters'):
                    event_filter = EventFilter(attributes=sub_config['filters'])
                
                sub = EventSubscription(
                    source=sub_config['source'],
                    events=events,
                    filters=event_filter,
                    transform=sub_config.get('transform')
                )
                # Add tier information as metadata
                sub.metadata['tier'] = sub_config.get('tier', 'standard')
                subscriptions.append(sub)
            
            self._external_subscriptions = subscriptions
        
        logger.info(f"📡 {self.container_id} configured external communication: {len(self._external_publications)} pubs, {len(self._external_subscriptions)} subs")
    
    def declare_external_publications(self, publications: List[EventPublication]) -> None:
        """Declare what events this container publishes externally"""
        self._external_publications = publications
    
    def declare_external_subscriptions(self, subscriptions: List[EventSubscription]) -> None:
        """Declare what events this container subscribes to externally"""
        self._external_subscriptions = subscriptions
    
    # === Router Management ===
    
    def unregister_from_router(self) -> None:
        """Unregister from event router during cleanup."""
        if self.external_router:
            # Check if router has unregister method
            if hasattr(self.external_router, 'unregister_container'):
                self.external_router.unregister_container(self.container_id)
            self.external_router = None
            logger.info(f"📡 {self.container_id} unregistered from Event Router")
    
    # === Utility Methods ===
    
    def get_communication_info(self) -> Dict[str, Any]:
        """Get information about this container's communication setup"""
        return {
            "container_id": self.container_id,
            "external_router_connected": self.external_router is not None,
            "external_publications": [
                {
                    "events": [str(e) for e in pub.events],
                    "scope": pub.scope.value,
                    "tier": getattr(pub, 'tier', 'standard')
                }
                for pub in self._external_publications
            ],
            "external_subscriptions": [
                {
                    "source": sub.source,
                    "events": [str(e) for e in sub.events],
                    "tier": getattr(sub, 'tier', 'standard'),
                    "has_filters": bool(sub.filters)
                }
                for sub in self._external_subscriptions
            ],
            "children_count": len(self.children),
            "has_parent": self.parent is not None,
            "internal_subscribers": len(self.internal_bus.all_subscribers)
        }
    
    def debug_event_flow(self) -> str:
        """Generate debug information about event flows"""
        info = self.get_communication_info()
        
        debug_str = f"\n=== {self.container_id} Communication Debug ===\n"
        debug_str += f"External Router: {'✅' if info['external_router_connected'] else '❌'}\n"
        debug_str += f"Publications: {len(info['external_publications'])}\n"
        debug_str += f"Subscriptions: {len(info['external_subscriptions'])}\n"
        debug_str += f"Children: {len(info['children_count'])}\n"
        debug_str += f"Parent: {'✅' if info['has_parent'] else '❌'}\n"
        
        if info['external_publications']:
            debug_str += "\nPublications:\n"
            for pub in info['external_publications']:
                debug_str += f"  • {pub['events']} → {pub['scope']} ({pub['tier']})\n"
        
        if info['external_subscriptions']:
            debug_str += "\nSubscriptions:\n"
            for sub in info['external_subscriptions']:
                debug_str += f"  • {sub['source']}.{sub['events']} ({sub['tier']})\n"
        
        return debug_str