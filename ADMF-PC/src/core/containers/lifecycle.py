"""
Container lifecycle management for ADMF-PC.

This module provides lifecycle management for containers, ensuring
proper initialization, execution, and cleanup of isolated execution
environments.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import threading
from datetime import datetime
import time

from .universal import UniversalScopedContainer, ContainerState, ContainerType


logger = logging.getLogger(__name__)


class LifecycleEvent(Enum):
    """Container lifecycle events."""
    CREATED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    STARTED = auto()
    STOPPING = auto()
    STOPPED = auto()
    RESETTING = auto()
    RESET = auto()
    DISPOSING = auto()
    DISPOSED = auto()
    FAILED = auto()


@dataclass
class ContainerInfo:
    """Information about a managed container."""
    container: UniversalScopedContainer
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContainerLifecycleManager:
    """
    Manages the lifecycle of multiple containers.
    
    This manager tracks all active containers, ensures proper cleanup,
    and provides centralized lifecycle control for the system.
    """
    
    def __init__(self, max_containers: Optional[int] = None):
        """
        Initialize the lifecycle manager.
        
        Args:
            max_containers: Maximum number of containers to manage
        """
        self.max_containers = max_containers
        self._containers: Dict[str, ContainerInfo] = {}
        self._lock = threading.RLock()
        
        # Lifecycle hooks
        self._lifecycle_hooks: Dict[LifecycleEvent, List[Callable]] = {
            event: [] for event in LifecycleEvent
        }
        
        # Container pools for reuse
        self._container_pools: Dict[str, List[UniversalScopedContainer]] = {}
        self._pool_sizes: Dict[str, int] = {}
        
        logger.info("ContainerLifecycleManager initialized")
    
    def create_container(
        self,
        container_type: str = "generic",
        container_id: Optional[str] = None,
        shared_services: Optional[Dict[str, Any]] = None,
        specs: Optional[List[Dict[str, Any]]] = None,
        initialize: bool = True,
        start: bool = False
    ) -> str:
        """
        Create and manage a new container.
        
        Args:
            container_type: Type of container
            container_id: Optional container ID
            shared_services: Services to share
            specs: Component specifications
            initialize: Whether to initialize immediately
            start: Whether to start after initialization
            
        Returns:
            Container ID
        """
        with self._lock:
            # Check container limit
            if self.max_containers and len(self._containers) >= self.max_containers:
                self._evict_oldest_container()
            
            # Try to get from pool first
            container = self._get_from_pool(container_type)
            
            if container:
                # Reset pooled container
                if container.state == ContainerState.STOPPED:
                    container.reset()
                container_id = container.container_id
                logger.info(f"Reusing pooled container: {container_id}")
            else:
                # Create new container
                container = UniversalScopedContainer(
                    container_id=container_id,
                    container_type=container_type,
                    shared_services=shared_services
                )
                container_id = container.container_id
                self._trigger_hooks(LifecycleEvent.CREATED, container)
            
            # Add components if specified
            if specs:
                for spec in specs:
                    container.create_component(spec)
            
            # Store container info
            self._containers[container_id] = ContainerInfo(container=container)
            
            # Initialize if requested
            if initialize:
                self.initialize_container(container_id)
                
                # Start if requested
                if start:
                    self.start_container(container_id)
            
            return container_id
    
    def initialize_container(self, container_id: str) -> None:
        """Initialize a container and all its components."""
        with self._lock:
            info = self._get_container_info(container_id)
            container = info.container
            
            if container.state != ContainerState.CREATED:
                raise RuntimeError(f"Container {container_id} already initialized")
            
            self._trigger_hooks(LifecycleEvent.INITIALIZING, container)
            
            try:
                container.initialize_scope()
                self._trigger_hooks(LifecycleEvent.INITIALIZED, container)
                
            except Exception as e:
                self._trigger_hooks(LifecycleEvent.FAILED, container)
                logger.error(f"Container {container_id} initialization failed: {e}")
                raise
    
    def start_container(self, container_id: str) -> None:
        """Start a container."""
        with self._lock:
            info = self._get_container_info(container_id)
            container = info.container
            
            self._trigger_hooks(LifecycleEvent.STARTING, container)
            
            try:
                container.start()
                self._trigger_hooks(LifecycleEvent.STARTED, container)
                
            except Exception as e:
                self._trigger_hooks(LifecycleEvent.FAILED, container)
                logger.error(f"Container {container_id} start failed: {e}")
                raise
    
    def stop_container(self, container_id: str) -> None:
        """Stop a container."""
        with self._lock:
            info = self._get_container_info(container_id)
            container = info.container
            
            self._trigger_hooks(LifecycleEvent.STOPPING, container)
            
            try:
                container.stop()
                self._trigger_hooks(LifecycleEvent.STOPPED, container)
                
            except Exception as e:
                logger.error(f"Container {container_id} stop failed: {e}")
                # Continue with cleanup even if stop fails
    
    def reset_container(self, container_id: str) -> None:
        """Reset a container to initialized state."""
        with self._lock:
            info = self._get_container_info(container_id)
            container = info.container
            
            self._trigger_hooks(LifecycleEvent.RESETTING, container)
            
            try:
                container.reset()
                self._trigger_hooks(LifecycleEvent.RESET, container)
                
            except Exception as e:
                logger.error(f"Container {container_id} reset failed: {e}")
                raise
    
    def dispose_container(
        self,
        container_id: str,
        return_to_pool: bool = False
    ) -> None:
        """
        Dispose of a container.
        
        Args:
            container_id: Container to dispose
            return_to_pool: Whether to return to pool for reuse
        """
        with self._lock:
            if container_id not in self._containers:
                return
            
            info = self._containers[container_id]
            container = info.container
            
            # Stop if running
            if container.state == ContainerState.RUNNING:
                self.stop_container(container_id)
            
            if return_to_pool and self._can_pool_container(container):
                # Return to pool for reuse
                self._return_to_pool(container)
                logger.info(f"Container {container_id} returned to pool")
            else:
                # Fully dispose
                self._trigger_hooks(LifecycleEvent.DISPOSING, container)
                container.dispose()
                self._trigger_hooks(LifecycleEvent.DISPOSED, container)
            
            # Remove from active containers
            del self._containers[container_id]
    
    def get_container(self, container_id: str) -> UniversalScopedContainer:
        """Get a container by ID."""
        with self._lock:
            info = self._get_container_info(container_id)
            info.last_accessed = datetime.now()
            info.access_count += 1
            return info.container
    
    def list_containers(
        self,
        container_type: Optional[str] = None,
        state: Optional[ContainerState] = None
    ) -> List[str]:
        """
        List containers matching criteria.
        
        Args:
            container_type: Filter by type
            state: Filter by state
            
        Returns:
            List of container IDs
        """
        with self._lock:
            results = []
            
            for container_id, info in self._containers.items():
                container = info.container
                
                # Apply filters
                if container_type and container.container_type != container_type:
                    continue
                    
                if state and container.state != state:
                    continue
                
                results.append(container_id)
            
            return results
    
    def dispose_all(self) -> None:
        """Dispose of all managed containers."""
        with self._lock:
            container_ids = list(self._containers.keys())
            
            for container_id in container_ids:
                try:
                    self.dispose_container(container_id)
                except Exception as e:
                    logger.error(f"Error disposing container {container_id}: {e}")
            
            # Clear pools
            for pool in self._container_pools.values():
                for container in pool:
                    try:
                        container.dispose()
                    except:
                        pass
            
            self._container_pools.clear()
    
    def add_lifecycle_hook(
        self,
        event: LifecycleEvent,
        hook: Callable[[UniversalScopedContainer], None]
    ) -> None:
        """Add a lifecycle hook."""
        self._lifecycle_hooks[event].append(hook)
    
    def configure_pool(
        self,
        container_type: str,
        pool_size: int
    ) -> None:
        """
        Configure container pooling.
        
        Args:
            container_type: Type of containers to pool
            pool_size: Maximum pool size
        """
        self._pool_sizes[container_type] = pool_size
        
        if container_type not in self._container_pools:
            self._container_pools[container_type] = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle manager statistics."""
        with self._lock:
            return {
                'active_containers': len(self._containers),
                'max_containers': self.max_containers,
                'containers_by_type': self._count_by_type(),
                'containers_by_state': self._count_by_state(),
                'pool_stats': {
                    container_type: {
                        'pooled': len(pool),
                        'max_size': self._pool_sizes.get(container_type, 0)
                    }
                    for container_type, pool in self._container_pools.items()
                },
                'total_created': sum(
                    info.access_count for info in self._containers.values()
                )
            }
    
    # Private methods
    
    def _get_container_info(self, container_id: str) -> ContainerInfo:
        """Get container info, raising if not found."""
        if container_id not in self._containers:
            raise ValueError(f"Container {container_id} not found")
        return self._containers[container_id]
    
    def _trigger_hooks(
        self,
        event: LifecycleEvent,
        container: UniversalScopedContainer
    ) -> None:
        """Trigger lifecycle hooks for an event."""
        for hook in self._lifecycle_hooks[event]:
            try:
                hook(container)
            except Exception as e:
                logger.error(f"Lifecycle hook error for {event}: {e}")
    
    def _evict_oldest_container(self) -> None:
        """Evict the oldest accessed container."""
        if not self._containers:
            return
        
        # Find oldest by last access time
        oldest_id = min(
            self._containers.keys(),
            key=lambda k: self._containers[k].last_accessed
        )
        
        logger.info(f"Evicting oldest container: {oldest_id}")
        self.dispose_container(oldest_id)
    
    def _can_pool_container(self, container: UniversalScopedContainer) -> bool:
        """Check if container can be pooled."""
        # Only pool stopped containers of pooled types
        if container.state != ContainerState.STOPPED:
            return False
        
        if container.container_type not in self._pool_sizes:
            return False
        
        # Check pool size limit
        pool = self._container_pools.get(container.container_type, [])
        max_size = self._pool_sizes.get(container.container_type, 0)
        
        return len(pool) < max_size
    
    def _get_from_pool(self, container_type: str) -> Optional[UniversalScopedContainer]:
        """Get a container from the pool."""
        pool = self._container_pools.get(container_type, [])
        
        if pool:
            return pool.pop()
        
        return None
    
    def _return_to_pool(self, container: UniversalScopedContainer) -> None:
        """Return a container to the pool."""
        container_type = container.container_type
        
        if container_type not in self._container_pools:
            self._container_pools[container_type] = []
        
        self._container_pools[container_type].append(container)
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count containers by type."""
        counts = {}
        
        for info in self._containers.values():
            container_type = info.container.container_type
            counts[container_type] = counts.get(container_type, 0) + 1
        
        return counts
    
    def _count_by_state(self) -> Dict[str, int]:
        """Count containers by state."""
        counts = {}
        
        for info in self._containers.values():
            state = info.container.state.name
            counts[state] = counts.get(state, 0) + 1
        
        return counts


# Global lifecycle manager instance
_lifecycle_manager = ContainerLifecycleManager()


def get_lifecycle_manager() -> ContainerLifecycleManager:
    """Get the global lifecycle manager."""
    return _lifecycle_manager