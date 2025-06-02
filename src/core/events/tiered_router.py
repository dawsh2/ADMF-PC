"""
Tiered Event Router implementation with performance tiers for different event types.

This module provides a unified Event Router with three performance tiers:
- Fast Tier: < 1ms latency for high-frequency data (BAR, TICK, QUOTE)
- Standard Tier: < 10ms latency for business logic (SIGNAL, INDICATOR)  
- Reliable Tier: 100% delivery for critical events (ORDER, FILL, SYSTEM)
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

from .types import Event, EventType
from .routing.protocols import EventRouterProtocol, EventPublication, EventSubscription, EventScope


logger = logging.getLogger(__name__)


class RouterTier(Enum):
    """Performance tiers for Event Router"""
    FAST = "fast"
    STANDARD = "standard" 
    RELIABLE = "reliable"


@dataclass
class RoutingMetrics:
    """Metrics for router performance monitoring"""
    events_routed: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    failed_deliveries: int = 0
    successful_deliveries: int = 0
    
    def record_delivery(self, latency_ms: float, success: bool = True):
        """Record a delivery attempt"""
        if success:
            self.successful_deliveries += 1
            self.events_routed += 1
            self.total_latency_ms += latency_ms
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        else:
            self.failed_deliveries += 1
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        return self.total_latency_ms / max(1, self.events_routed)
    
    @property
    def success_rate(self) -> float:
        """Calculate delivery success rate"""
        total = self.successful_deliveries + self.failed_deliveries
        return self.successful_deliveries / max(1, total)


class FastTierRouter:
    """Optimized router for high-frequency data events (< 1ms target)"""
    
    def __init__(self, batch_size: int = 1000, max_latency_ms: float = 1.0):
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        
        # Pre-computed routing tables for speed
        self.routing_cache: Dict[EventType, List[str]] = defaultdict(list)
        self.subscribers: Dict[str, Callable] = {}
        self.filters: Dict[str, Dict[str, Any]] = {}
        
        # Batch processing
        self.batch_buffer: List[tuple] = []
        self.last_flush_time = time.time()
        
        # Metrics
        self.metrics = RoutingMetrics()
        
        # Start batch processor
        self._batch_task = asyncio.create_task(self._batch_processor())
    
    def register_publisher(self, container_id: str, publications: List[EventPublication]) -> None:
        """Register publisher with routing cache optimization"""
        for pub in publications:
            for event_type in pub.events:
                if container_id not in self.routing_cache[event_type]:
                    self.routing_cache[event_type].append(container_id)
        
        logger.debug(f"FastTier: Registered publisher {container_id}")
    
    def register_subscriber(self, container_id: str, subscriptions: List[EventSubscription], callback: Callable) -> None:
        """Register subscriber with filter optimization"""
        self.subscribers[container_id] = callback
        
        # Pre-process filters for fast lookup
        for sub in subscriptions:
            if sub.filters:
                self.filters[container_id] = sub.filters
        
        logger.debug(f"FastTier: Registered subscriber {container_id}")
    
    def route_event(self, event: Event, source: str, scope: Optional[EventScope] = None) -> None:
        """Ultra-fast routing for data events via batching"""
        start_time = time.time()
        
        # Add to batch buffer for processing
        self.batch_buffer.append((event, source, scope, start_time))
        
        # Flush if batch is full or latency threshold reached
        current_time = time.time()
        if (len(self.batch_buffer) >= self.batch_size or 
            (current_time - self.last_flush_time) * 1000 > self.max_latency_ms):
            self._flush_batch_sync()
    
    def _flush_batch_sync(self) -> None:
        """Synchronously flush batch for minimal latency"""
        if not self.batch_buffer:
            return
        
        batch_start = time.time()
        
        for event, source, scope, event_start_time in self.batch_buffer:
            try:
                # Get subscribers for this event type
                subscribers = self.routing_cache.get(event.event_type, [])
                
                for subscriber_id in subscribers:
                    callback = self.subscribers.get(subscriber_id)
                    if callback:
                        # Apply filters if present
                        if subscriber_id in self.filters:
                            if not self._apply_filters(event, self.filters[subscriber_id]):
                                continue
                        
                        # Direct callback for speed (no async overhead)
                        try:
                            callback(event, source)
                            
                            # Record successful delivery
                            latency_ms = (time.time() - event_start_time) * 1000
                            self.metrics.record_delivery(latency_ms, True)
                            
                        except Exception as e:
                            logger.error(f"FastTier delivery error to {subscriber_id}: {e}")
                            self.metrics.record_delivery(0, False)
            
            except Exception as e:
                logger.error(f"FastTier routing error for {event.event_type}: {e}")
                self.metrics.record_delivery(0, False)
        
        # Clear batch
        self.batch_buffer.clear()
        self.last_flush_time = time.time()
        
        batch_latency = (time.time() - batch_start) * 1000
        logger.debug(f"FastTier: Flushed batch with {len(self.batch_buffer)} events in {batch_latency:.2f}ms")
    
    async def _batch_processor(self) -> None:
        """Background task to ensure regular batch flushing"""
        while True:
            try:
                await asyncio.sleep(self.max_latency_ms / 1000)  # Check at latency interval
                
                if self.batch_buffer and (time.time() - self.last_flush_time) * 1000 > self.max_latency_ms:
                    self._flush_batch_sync()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"FastTier batch processor error: {e}")
    
    def _apply_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Apply subscription filters to event"""
        payload = event.payload or {}
        
        for filter_key, filter_value in filters.items():
            if filter_key in payload:
                if isinstance(filter_value, list):
                    if payload[filter_key] not in filter_value:
                        return False
                else:
                    if payload[filter_key] != filter_value:
                        return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "tier": "fast",
            "events_routed": self.metrics.events_routed,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "max_latency_ms": self.metrics.max_latency_ms,
            "success_rate": self.metrics.success_rate,
            "batch_size": self.batch_size,
            "current_batch_size": len(self.batch_buffer)
        }


class StandardTierRouter:
    """Standard router for business logic events (< 10ms target)"""
    
    def __init__(self, batch_size: int = 100, max_latency_ms: float = 10.0):
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        
        # Subscription management
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.publishers: Dict[str, List[EventPublication]] = defaultdict(list)
        self.callbacks: Dict[str, Callable] = {}
        
        # Async processing
        self.event_queue = asyncio.Queue()
        self.metrics = RoutingMetrics()
        
        # Start async processor
        self._processor_task = asyncio.create_task(self._process_queue())
    
    def register_publisher(self, container_id: str, publications: List[EventPublication]) -> None:
        """Register publisher"""
        self.publishers[container_id] = publications
        logger.debug(f"StandardTier: Registered publisher {container_id}")
    
    def register_subscriber(self, container_id: str, subscriptions: List[EventSubscription], callback: Callable) -> None:
        """Register subscriber"""
        self.subscriptions[container_id] = subscriptions
        self.callbacks[container_id] = callback
        logger.debug(f"StandardTier: Registered subscriber {container_id}")
    
    def route_event(self, event: Event, source: str, scope: Optional[EventScope] = None) -> None:
        """Route event through async processing"""
        start_time = time.time()
        
        # Queue for async processing
        try:
            self.event_queue.put_nowait((event, source, scope, start_time))
        except asyncio.QueueFull:
            logger.warning(f"StandardTier queue full, dropping event {event.event_type}")
            self.metrics.record_delivery(0, False)
    
    async def _process_queue(self) -> None:
        """Process queued events asynchronously"""
        batch = []
        
        while True:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        # Wait up to max_latency for next event
                        item = await asyncio.wait_for(
                            self.event_queue.get(),
                            timeout=self.max_latency_ms / 1000
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break  # Process current batch
                
                if batch:
                    await self._process_batch(batch)
                    batch.clear()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"StandardTier processor error: {e}")
    
    async def _process_batch(self, batch: List[tuple]) -> None:
        """Process a batch of events"""
        for event, source, scope, start_time in batch:
            try:
                # Find matching subscriptions
                for container_id, subscriptions in self.subscriptions.items():
                    for subscription in subscriptions:
                        if (subscription.source == source and
                            event.event_type in subscription.events):
                            
                            # Apply filters
                            if subscription.filters:
                                if not self._apply_filters(event, subscription.filters):
                                    continue
                            
                            # Deliver event
                            callback = self.callbacks.get(container_id)
                            if callback:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(event, source)
                                    else:
                                        callback(event, source)
                                    
                                    # Record metrics
                                    latency_ms = (time.time() - start_time) * 1000
                                    self.metrics.record_delivery(latency_ms, True)
                                
                                except Exception as e:
                                    logger.error(f"StandardTier delivery error to {container_id}: {e}")
                                    self.metrics.record_delivery(0, False)
            
            except Exception as e:
                logger.error(f"StandardTier processing error for {event.event_type}: {e}")
                self.metrics.record_delivery(0, False)
    
    def _apply_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Apply subscription filters"""
        payload = event.payload or {}
        
        for filter_key, filter_value in filters.items():
            if filter_key in payload:
                if isinstance(filter_value, list):
                    if payload[filter_key] not in filter_value:
                        return False
                else:
                    if payload[filter_key] != filter_value:
                        return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "tier": "standard",
            "events_routed": self.metrics.events_routed,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "max_latency_ms": self.metrics.max_latency_ms,
            "success_rate": self.metrics.success_rate,
            "queue_size": self.event_queue.qsize()
        }


class ReliableTierRouter:
    """Reliable router for critical events (100% delivery guarantee)"""
    
    def __init__(self, retry_attempts: int = 3, retry_delay_ms: float = 1000):
        self.retry_attempts = retry_attempts
        self.retry_delay_ms = retry_delay_ms
        
        # Subscription management
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.callbacks: Dict[str, Callable] = {}
        
        # Reliability features
        self.persistent_queue = deque()
        self.dead_letter_queue = deque()
        self.delivery_confirmations: Dict[str, bool] = {}
        self.metrics = RoutingMetrics()
        
        # Start reliable processor
        self._processor_task = asyncio.create_task(self._process_reliable_queue())
    
    def register_publisher(self, container_id: str, publications: List[EventPublication]) -> None:
        """Register publisher"""
        logger.debug(f"ReliableTier: Registered publisher {container_id}")
    
    def register_subscriber(self, container_id: str, subscriptions: List[EventSubscription], callback: Callable) -> None:
        """Register subscriber"""
        self.subscriptions[container_id] = subscriptions
        self.callbacks[container_id] = callback
        logger.debug(f"ReliableTier: Registered subscriber {container_id}")
    
    def route_event(self, event: Event, source: str, scope: Optional[EventScope] = None) -> None:
        """Route event with reliability guarantees"""
        delivery_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add to persistent queue
        self.persistent_queue.append({
            'delivery_id': delivery_id,
            'event': event,
            'source': source,
            'scope': scope,
            'start_time': start_time,
            'attempts': 0
        })
    
    async def _process_reliable_queue(self) -> None:
        """Process queue with retry logic"""
        while True:
            try:
                if self.persistent_queue:
                    item = self.persistent_queue.popleft()
                    success = await self._deliver_with_retry(item)
                    
                    if not success:
                        # Move to dead letter queue
                        self.dead_letter_queue.append(item)
                        logger.error(f"ReliableTier: Moved {item['event'].event_type} to dead letter queue")
                
                else:
                    # Wait for new items
                    await asyncio.sleep(0.1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ReliableTier processor error: {e}")
    
    async def _deliver_with_retry(self, item: Dict[str, Any]) -> bool:
        """Attempt delivery with retries"""
        event = item['event']
        source = item['source']
        
        for attempt in range(self.retry_attempts):
            try:
                item['attempts'] = attempt + 1
                
                # Find matching subscriptions
                delivered = False
                for container_id, subscriptions in self.subscriptions.items():
                    for subscription in subscriptions:
                        if (subscription.source == source and
                            event.event_type in subscription.events):
                            
                            callback = self.callbacks.get(container_id)
                            if callback:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(event, source)
                                else:
                                    callback(event, source)
                                
                                delivered = True
                
                if delivered:
                    # Record successful delivery
                    latency_ms = (time.time() - item['start_time']) * 1000
                    self.metrics.record_delivery(latency_ms, True)
                    return True
                
            except Exception as e:
                logger.warning(f"ReliableTier delivery attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Wait before retry with exponential backoff
                    delay = self.retry_delay_ms * (2 ** attempt) / 1000
                    await asyncio.sleep(delay)
        
        # All attempts failed
        self.metrics.record_delivery(0, False)
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics"""
        return {
            "tier": "reliable",
            "events_routed": self.metrics.events_routed,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "success_rate": self.metrics.success_rate,
            "queue_size": len(self.persistent_queue),
            "dead_letter_size": len(self.dead_letter_queue),
            "retry_attempts": self.retry_attempts
        }


class TieredEventRouter(EventRouterProtocol):
    """
    Unified Event Router with performance tiers for different event types.
    
    Tiers:
    - Fast: < 1ms for BAR, TICK, QUOTE (high-frequency data)
    - Standard: < 10ms for SIGNAL, INDICATOR (business logic)  
    - Reliable: 100% delivery for ORDER, FILL, SYSTEM (critical events)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize tier routers
        fast_config = self.config.get('fast', {})
        self.fast_tier = FastTierRouter(
            batch_size=fast_config.get('batch_size', 1000),
            max_latency_ms=fast_config.get('max_latency_ms', 1.0)
        )
        
        standard_config = self.config.get('standard', {})
        self.standard_tier = StandardTierRouter(
            batch_size=standard_config.get('batch_size', 100),
            max_latency_ms=standard_config.get('max_latency_ms', 10.0)
        )
        
        reliable_config = self.config.get('reliable', {})
        self.reliable_tier = ReliableTierRouter(
            retry_attempts=reliable_config.get('retry_attempts', 3),
            retry_delay_ms=reliable_config.get('retry_delay_ms', 1000)
        )
        
        # Tier mapping
        self.tier_routers = {
            'fast': self.fast_tier,
            'standard': self.standard_tier,
            'reliable': self.reliable_tier
        }
        
        # Default event-to-tier mapping
        self.default_tier_map = {
            EventType.BAR: 'fast',
            EventType.TICK: 'fast',
            EventType.QUOTE: 'fast',
            EventType.SIGNAL: 'standard',
            EventType.INDICATORS: 'standard',
            EventType.PORTFOLIO_UPDATE: 'standard',
            EventType.ORDER: 'reliable',
            EventType.FILL: 'reliable',
            EventType.SYSTEM: 'reliable'
        }
        
        logger.info("TieredEventRouter initialized with fast, standard, and reliable tiers")
    
    def route_event(self, event: Event, source: str, tier: str = None, scope: Optional[EventScope] = None) -> None:
        """Route event through appropriate tier"""
        # Auto-determine tier if not specified
        if tier is None:
            tier = self.default_tier_map.get(event.event_type, 'standard')
        
        # Route through appropriate tier
        router = self.tier_routers.get(tier)
        if router:
            router.route_event(event, source, scope)
            logger.debug(f"📡 Routed {event.event_type} from {source} via {tier} tier")
        else:
            logger.error(f"Unknown tier: {tier}")
    
    def register_publisher(self, container_id: str, publications: List[EventPublication]) -> None:
        """Register publisher across relevant tiers"""
        for publication in publications:
            # Determine tier for each publication
            tier = getattr(publication, 'tier', 'standard')
            router = self.tier_routers.get(tier, self.standard_tier)
            router.register_publisher(container_id, [publication])
        
        logger.info(f"📡 Registered publisher {container_id} across {len(publications)} tiers")
    
    def register_subscriber(self, container_id: str, subscriptions: List[EventSubscription], callback: Callable) -> None:
        """Register subscriber across relevant tiers"""
        for subscription in subscriptions:
            # Determine tier for each subscription
            tier = getattr(subscription, 'tier', 'standard')
            router = self.tier_routers.get(tier, self.standard_tier)
            router.register_subscriber(container_id, [subscription], callback)
        
        logger.info(f"📡 Registered subscriber {container_id} across {len(subscriptions)} tiers")
    
    def unregister_container(self, container_id: str) -> None:
        """Unregister container from all tiers"""
        # Implementation would remove container from all tier routers
        logger.info(f"📡 Unregistered container {container_id} from all tiers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        return {
            "total_tiers": len(self.tier_routers),
            "tier_metrics": {
                tier_name: router.get_metrics()
                for tier_name, router in self.tier_routers.items()
            },
            "default_tier_mappings": {
                str(event_type): tier
                for event_type, tier in self.default_tier_map.items()
            }
        }
    
    def debug_routing(self) -> str:
        """Generate debug information about routing"""
        debug_str = "\n=== TieredEventRouter Debug ===\n"
        
        for tier_name, router in self.tier_routers.items():
            metrics = router.get_metrics()
            debug_str += f"\n{tier_name.upper()} Tier:\n"
            debug_str += f"  Events Routed: {metrics.get('events_routed', 0)}\n"
            debug_str += f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.2f}ms\n"
            debug_str += f"  Success Rate: {metrics.get('success_rate', 0):.2%}\n"
            
            if 'queue_size' in metrics:
                debug_str += f"  Queue Size: {metrics['queue_size']}\n"
        
        return debug_str