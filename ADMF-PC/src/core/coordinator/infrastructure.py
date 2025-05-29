"""
Infrastructure management for the Coordinator.
"""
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import logging

from ..infrastructure import InfrastructureCapability
from ..events import EventBus
from ..containers import UniversalContainer
from .types import WorkflowConfig, ExecutionContext


logger = logging.getLogger(__name__)


@dataclass
class SharedResource:
    """Represents a shared resource."""
    resource_id: str
    resource_type: str
    instance: Any
    users: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InfrastructureSetup:
    """Manages infrastructure setup for workflows."""
    
    def __init__(
        self,
        container: UniversalContainer,
        event_bus: EventBus
    ):
        self.container = container
        self.event_bus = event_bus
        self._shared_resources: Dict[str, SharedResource] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def setup_workflow_infrastructure(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Set up infrastructure for a workflow."""
        infrastructure = {}
        
        # Set up data feeds
        if config.data_config:
            data_feeds = await self._setup_data_feeds(config.data_config)
            infrastructure['data_feeds'] = data_feeds
            
        # Set up shared indicators
        if config.infrastructure_config.get('indicators'):
            indicators = await self._setup_shared_indicators(
                config.infrastructure_config['indicators']
            )
            infrastructure['indicators'] = indicators
            
        # Set up computation resources
        if config.infrastructure_config.get('computation'):
            computation = await self._setup_computation_resources(
                config.infrastructure_config['computation']
            )
            infrastructure['computation'] = computation
            
        # Store in context
        context.shared_resources.update(infrastructure)
        
        # Emit setup complete event
        await self.event_bus.emit({
            'type': 'infrastructure.setup.complete',
            'workflow_id': context.workflow_id,
            'resources': list(infrastructure.keys())
        })
        
        return infrastructure
        
    async def teardown_workflow_infrastructure(
        self,
        context: ExecutionContext
    ) -> None:
        """Tear down infrastructure for a workflow."""
        # Release shared resources
        for resource_type, resources in context.shared_resources.items():
            if isinstance(resources, dict):
                for resource_id in resources:
                    await self._release_shared_resource(
                        resource_id,
                        context.workflow_id
                    )
                    
        # Emit teardown event
        await self.event_bus.emit({
            'type': 'infrastructure.teardown.complete',
            'workflow_id': context.workflow_id
        })
        
    async def _setup_data_feeds(
        self,
        data_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up data feeds."""
        feeds = {}
        
        for feed_name, feed_config in data_config.items():
            # Check if feed already exists
            resource_id = f"data_feed:{feed_name}"
            
            async with self._get_lock(resource_id):
                if resource_id in self._shared_resources:
                    # Reuse existing feed
                    feed = self._shared_resources[resource_id].instance
                else:
                    # Create new feed
                    feed = await self._create_data_feed(feed_name, feed_config)
                    self._shared_resources[resource_id] = SharedResource(
                        resource_id=resource_id,
                        resource_type='data_feed',
                        instance=feed,
                        metadata=feed_config
                    )
                    
                feeds[feed_name] = feed
                
        return feeds
        
    async def _setup_shared_indicators(
        self,
        indicator_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up shared indicators."""
        indicators = {}
        
        for indicator_name, config in indicator_config.items():
            resource_id = f"indicator:{indicator_name}"
            
            async with self._get_lock(resource_id):
                if resource_id in self._shared_resources:
                    # Reuse existing indicator
                    indicator = self._shared_resources[resource_id].instance
                else:
                    # Create new indicator
                    indicator = await self._create_indicator(
                        indicator_name,
                        config
                    )
                    self._shared_resources[resource_id] = SharedResource(
                        resource_id=resource_id,
                        resource_type='indicator',
                        instance=indicator,
                        metadata=config
                    )
                    
                indicators[indicator_name] = indicator
                
        return indicators
        
    async def _setup_computation_resources(
        self,
        computation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up computation resources."""
        resources = {}
        
        # Set up thread pools
        if computation_config.get('thread_pool'):
            pool_config = computation_config['thread_pool']
            resources['thread_pool'] = await self._create_thread_pool(
                pool_config
            )
            
        # Set up process pools
        if computation_config.get('process_pool'):
            pool_config = computation_config['process_pool']
            resources['process_pool'] = await self._create_process_pool(
                pool_config
            )
            
        # Set up GPU resources
        if computation_config.get('gpu'):
            gpu_config = computation_config['gpu']
            resources['gpu'] = await self._allocate_gpu_resources(
                gpu_config
            )
            
        return resources
        
    async def _create_data_feed(
        self,
        feed_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """Create a data feed instance."""
        # This would integrate with your data system
        logger.info(f"Creating data feed: {feed_name}")
        
        # Get data handler from container
        data_handler = await self.container.get('data_handler')
        
        # Create feed configuration
        feed = await data_handler.create_feed(feed_name, config)
        
        return feed
        
    async def _create_indicator(
        self,
        indicator_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """Create an indicator instance."""
        logger.info(f"Creating indicator: {indicator_name}")
        
        # This would integrate with your indicator system
        # For now, return a placeholder
        return {'name': indicator_name, 'config': config}
        
    async def _create_thread_pool(
        self,
        config: Dict[str, Any]
    ) -> Any:
        """Create a thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        
        max_workers = config.get('max_workers', 4)
        return ThreadPoolExecutor(max_workers=max_workers)
        
    async def _create_process_pool(
        self,
        config: Dict[str, Any]
    ) -> Any:
        """Create a process pool."""
        from concurrent.futures import ProcessPoolExecutor
        
        max_workers = config.get('max_workers', 4)
        return ProcessPoolExecutor(max_workers=max_workers)
        
    async def _allocate_gpu_resources(
        self,
        config: Dict[str, Any]
    ) -> Any:
        """Allocate GPU resources."""
        logger.info(f"Allocating GPU resources: {config}")
        
        # This would integrate with GPU management
        # For now, return a placeholder
        return {'gpu_config': config}
        
    async def _release_shared_resource(
        self,
        resource_id: str,
        user_id: str
    ) -> None:
        """Release a shared resource."""
        async with self._get_lock(resource_id):
            if resource_id in self._shared_resources:
                resource = self._shared_resources[resource_id]
                resource.users.discard(user_id)
                
                # If no more users, clean up
                if not resource.users:
                    logger.info(f"Cleaning up resource: {resource_id}")
                    
                    # Cleanup based on resource type
                    if hasattr(resource.instance, 'close'):
                        await resource.instance.close()
                    elif hasattr(resource.instance, 'shutdown'):
                        resource.instance.shutdown()
                        
                    del self._shared_resources[resource_id]
                    
    def _get_lock(self, resource_id: str) -> asyncio.Lock:
        """Get or create a lock for a resource."""
        if resource_id not in self._locks:
            self._locks[resource_id] = asyncio.Lock()
        return self._locks[resource_id]