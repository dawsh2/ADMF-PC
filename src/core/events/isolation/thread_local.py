"""Event isolation for parallel execution."""

import threading
from typing import Dict, Optional, Any
import weakref

from ..bus import EventBus

class EventIsolationManager:
    """
    Ensures complete isolation between parallel executions.
    
    Critical for parallel backtesting and optimization.
    """
    
    _thread_local = threading.local()
    _container_buses: Dict[str, weakref.ref] = {}
    
    @classmethod
    def get_isolated_bus(cls, container_id: str) -> EventBus:
        """Get thread-local isolated event bus for container."""
        # Ensure thread has its own bus registry
        if not hasattr(cls._thread_local, 'buses'):
            cls._thread_local.buses = {}
            
        # Get or create bus for this container in this thread
        if container_id not in cls._thread_local.buses:
            bus = EventBus(f"{container_id}_thread_{threading.get_ident()}")
            cls._thread_local.buses[container_id] = bus
            
            # Store weak reference for cleanup
            cls._container_buses[f"{container_id}_{threading.get_ident()}"] = \
                weakref.ref(bus)
                
        return cls._thread_local.buses[container_id]
    
    @classmethod
    def cleanup_thread(cls) -> None:
        """Clean up all buses for current thread."""
        if hasattr(cls._thread_local, 'buses'):
            cls._thread_local.buses.clear()
            
    @classmethod
    def cleanup_container(cls, container_id: str) -> None:
        """Clean up bus for specific container in current thread."""
        if hasattr(cls._thread_local, 'buses'):
            if container_id in cls._thread_local.buses:
                del cls._thread_local.buses[container_id]
                
    @classmethod
    def get_active_buses(cls) -> Dict[str, int]:
        """Get count of active buses by container (for monitoring)."""
        active = {}
        
        # Clean up dead references
        dead_keys = []
        for key, ref in cls._container_buses.items():
            if ref() is None:
                dead_keys.append(key)
            else:
                container_id = key.split('_thread_')[0]
                active[container_id] = active.get(container_id, 0) + 1
                
        # Remove dead references
        for key in dead_keys:
            del cls._container_buses[key]
            
        return active
