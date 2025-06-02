"""
Bootstrap script for setting up the Hybrid Tiered Communication system.

This module handles the initialization of the TieredEventRouter and 
registration of all containers for cross-container communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from ..containers.composable import ComposableContainerProtocol
from .tiered_router import TieredEventRouter
from .hybrid_interface import HybridContainerInterface


logger = logging.getLogger(__name__)


async def setup_hybrid_communication(
    root_container: ComposableContainerProtocol,
    router_config: Optional[Dict[str, Any]] = None
) -> TieredEventRouter:
    """
    Set up the Hybrid Tiered Communication system.
    
    This function:
    1. Creates and configures the TieredEventRouter
    2. Recursively registers all containers in the hierarchy
    3. Sets up automatic communication routing
    
    Args:
        root_container: Root container of the system
        router_config: Optional configuration for the TieredEventRouter
        
    Returns:
        Configured TieredEventRouter instance
    """
    logger.info("🚀 Setting up Hybrid Tiered Communication system")
    
    # Create TieredEventRouter with configuration
    router = TieredEventRouter(config=router_config or {})
    
    # Register all containers in the hierarchy
    await register_container_hierarchy(root_container, router)
    
    # Log setup completion
    stats = router.get_stats()
    logger.info(f"✅ Hybrid Communication setup complete: {stats}")
    
    return router


async def register_container_hierarchy(
    container: ComposableContainerProtocol,
    router: TieredEventRouter
) -> None:
    """
    Recursively register all containers in the hierarchy with the Event Router.
    
    This ensures that all containers, including dynamically created sub-containers,
    are properly registered for cross-container communication.
    """
    containers_registered = []
    
    def register_container_tree(cont):
        """Recursively register containers"""
        if hasattr(cont, 'register_with_router'):
            cont.register_with_router(router)
            containers_registered.append(cont.container_id)
            logger.debug(f"📡 Registered {cont.container_id} with Event Router")
        
        # Recursively register children
        for child in getattr(cont, 'child_containers', []):
            register_container_tree(child)
    
    # Register the entire tree
    register_container_tree(container)
    
    logger.info(f"📡 Registered {len(containers_registered)} containers with Event Router")
    
    # Debug: Log container communication info
    if logger.isEnabledFor(logging.DEBUG):
        for cont in _get_all_containers(container):
            if hasattr(cont, 'debug_event_flow'):
                logger.debug(cont.debug_event_flow())


def _get_all_containers(root_container: ComposableContainerProtocol) -> List[ComposableContainerProtocol]:
    """Get all containers in the hierarchy"""
    containers = []
    
    def collect_containers(container):
        containers.append(container)
        for child in getattr(container, 'child_containers', []):
            collect_containers(child)
    
    collect_containers(root_container)
    return containers


async def validate_communication_setup(
    root_container: ComposableContainerProtocol,
    router: TieredEventRouter
) -> Dict[str, Any]:
    """
    Validate that the communication setup is correct.
    
    Returns:
        Dictionary with validation results and any issues found
    """
    logger.info("🔍 Validating communication setup")
    
    issues = []
    warnings = []
    containers_checked = 0
    
    # Get all containers
    all_containers = _get_all_containers(root_container)
    
    for container in all_containers:
        containers_checked += 1
        
        # Check if hybrid interface is implemented
        if not hasattr(container, 'register_with_router'):
            issues.append(f"Container {getattr(container, 'container_id', 'unknown')} missing hybrid interface")
            continue
        
        # Check if registered with router
        if not hasattr(container, 'external_router') or container.external_router is None:
            warnings.append(f"Container {container.container_id} not registered with router")
        
        # Check communication configuration
        if hasattr(container, 'get_communication_info'):
            comm_info = container.get_communication_info()
            
            # Validate publications
            if not comm_info.get('external_publications'):
                if hasattr(container, '_external_publications') and container._external_publications:
                    warnings.append(f"Container {container.container_id} has publications but not connected to router")
            
            # Validate subscriptions
            if not comm_info.get('external_subscriptions'):
                if hasattr(container, '_external_subscriptions') and container._external_subscriptions:
                    warnings.append(f"Container {container.container_id} has subscriptions but not connected to router")
    
    # Check router statistics
    router_stats = router.get_stats()
    
    validation_result = {
        'status': 'valid' if not issues else 'invalid',
        'containers_checked': containers_checked,
        'issues': issues,
        'warnings': warnings,
        'router_stats': router_stats
    }
    
    if issues:
        logger.error(f"❌ Communication validation failed: {len(issues)} issues found")
        for issue in issues:
            logger.error(f"  • {issue}")
    else:
        logger.info(f"✅ Communication validation passed")
    
    if warnings:
        logger.warning(f"⚠️ {len(warnings)} warnings found:")
        for warning in warnings:
            logger.warning(f"  • {warning}")
    
    return validation_result


async def demonstrate_communication_flow(
    router: TieredEventRouter,
    container_hierarchy: ComposableContainerProtocol
) -> None:
    """
    Demonstrate the communication flow by sending test events.
    
    This is useful for debugging and understanding the communication patterns.
    """
    logger.info("🧪 Demonstrating communication flow")
    
    from .types import Event, EventType
    from datetime import datetime
    
    # Create test events for each tier
    test_events = [
        {
            'name': 'Fast Tier BAR Event',
            'event': Event(
                event_type=EventType.BAR,
                payload={
                    'symbol': 'TEST',
                    'timestamp': datetime.now(),
                    'market_data': {'TEST': {'close': 100.0}}
                },
                timestamp=datetime.now()
            ),
            'tier': 'fast'
        },
        {
            'name': 'Standard Tier SIGNAL Event',
            'event': Event(
                event_type=EventType.SIGNAL,
                payload={
                    'signals': [],
                    'timestamp': datetime.now()
                },
                timestamp=datetime.now()
            ),
            'tier': 'standard'
        },
        {
            'name': 'Reliable Tier SYSTEM Event',
            'event': Event(
                event_type=EventType.SYSTEM,
                payload={
                    'action': 'TEST',
                    'timestamp': datetime.now()
                },
                timestamp=datetime.now()
            ),
            'tier': 'reliable'
        }
    ]
    
    # Send test events through each tier
    for test in test_events:
        logger.info(f"📤 Sending {test['name']} via {test['tier']} tier")
        
        try:
            router.route_event(test['event'], 'test_source', test['tier'])
            logger.info(f"✅ {test['name']} sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send {test['name']}: {e}")
    
    # Wait for events to process
    await asyncio.sleep(0.1)
    
    # Display router metrics
    logger.info("📊 Router metrics after test:")
    logger.info(router.debug_routing())


def get_default_router_config() -> Dict[str, Any]:
    """Get default configuration for TieredEventRouter"""
    return {
        'fast': {
            'batch_size': 1000,
            'max_latency_ms': 1.0
        },
        'standard': {
            'batch_size': 100,
            'max_latency_ms': 10.0
        },
        'reliable': {
            'retry_attempts': 3,
            'retry_delay_ms': 1000
        }
    }